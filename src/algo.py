import torch

from rlpyt.utils.collections import namedarraytuple
from collections import namedtuple
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.logging import logger
from src.models import *
from torch.utils.tensorboard import SummaryWriter
import time

Samples = namedtuple("Samples",
    ["observation", "action", "reward", "done"])

def iter_loader(loader_list):
    for loader in loader_list:
        for b in loader:
            yield b

def tensor2list(t:torch.Tensor):
    #:param: t:(t,...)
    length =  t.shape[0]
    t_list = []
    for i in range(length):
        t_list.append(t[i])
    return  t_list

class DQNSPR():
    def __init__(self, train_loaders, valid_loaders):
        #init optimizer
        self.device = 'cuda'
        self.horizon = 10
        self.model = SPRDqnModel(horizon = self.horizon).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.n_step_return = 1
        self.discount = 0.99
        
        self.config_optimizer()
        self.writer = SummaryWriter(comment=f'train_{self.horizon}', max_queue=500, flush_secs=600)
        self.valid_writer = SummaryWriter(comment=f'valid_{self.horizon}', max_queue=50, flush_secs=600)
        self.fake_writer = SummaryWriter(comment='fake', max_queue=500, flush_secs=600)
        self.train_loaders = train_loaders
        self.valid_loaders = iter_loader(valid_loaders)
        self.iter = 0
        
    
    def config_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4)

        self.gan_optimizer = torch.optim.Adam(self.discriminator.parameters(),lr=1e-4)

    def dqn_rl_loss(self, qs, samples, index):
        """
        Computes the Q-learning loss, based on: 0.5 * (Q - target_Q) ^ 2.
        Implements regular DQN or Double-DQN for computing target_Q values
        using the agent's target network.  Computes the Huber loss using
        ``delta_clip``, or if ``None``, uses MSE.  When using prioritized
        replay, multiplies losses by importance sample weights.

        Input ``samples`` have leading batch dimension [B,..] (but not time).

        Calls the agent to compute forward pass on training inputs, and calls
        ``agent.target()`` to compute target values.

        Returns loss and TD-absolute-errors for use in prioritization.

        Warning:
            If not using mid_batch_reset, the sampler will only reset environments
            between iterations, so some samples in the replay buffer will be
            invalid.  This case is not supported here currently.
        """
        #:param: samples: namedtuple of torch.Tensor ((t,b,..))
        #:param: index: int: the TD-error is calculated from sample[index]
        #:param: qs: (bz, act_dim)
        #:return: losses: (bz)

        indices = samples.action[index+1].squeeze()
        
        q = select_at_indexes(indices, qs)
        with torch.no_grad():
            target_qs = self.model.target_q(samples.observation[index + self.n_step_return])
            target_q = torch.max(target_qs, dim=-1).values
            disc_target_q = (self.discount ** self.n_step_return) * target_q
            #bz
            y = samples.reward[index].squeeze() + (1 - samples.done[index].float().squeeze()) * disc_target_q
        delta = y - q
        losses = 0.5 * delta ** 2
        return losses 
    
    def log_twin_sequence(self, pred, origin, train=True):
        #:param: pred:(t,4,84,84)
        #:param: origin:(t,4,84,84)
        
        length = pred.shape[0]
        
        pred_sequence = torch.cat(tensor2list(pred),dim=2)
        origin_sequence = torch.cat(tensor2list(origin),dim=2)
        #(4,84,84*t)
        assert pred_sequence.shape[0]==4
        assert pred_sequence.shape[1]==84
        full_sequence = torch.cat([pred_sequence[:3], origin_sequence[:3]],dim=1)
        if train:
            self.writer.add_image('Model/pred_sequence', full_sequence, self.iter)
        else:
            self.valid_writer.add_image('Model/pred_sequence', full_sequence, self.iter)

    def compute_gan_loss(self, pred_obss, observation):
        #:param: pred_obss:(t,b,c,84,84)
        #:param: observation:(t,b,c,84,84)
        target_obs = self.model.transform(observation, False)
        length, batch_size = pred_obss.shape[:2]
        labels = torch.cat((torch.ones(length*batch_size, dtype=torch.long, device=self.device), \
        torch.zeros(length*batch_size, dtype=torch.long, device=self.device)), dim=0)

        pred_origin = self.discriminator(target_obs.view(length*batch_size, *target_obs.shape[2:]))
        pred_target = self.discriminator(pred_obss.view(length*batch_size, *pred_obss.shape[2:]))
        pred_origin_v2 = self.discriminator(target_obs.view(length*batch_size, *target_obs.shape[2:]).detach())
        pred_target_v2 = self.discriminator(pred_obss.view(length*batch_size, *pred_obss.shape[2:]).detach())
        #(t*b,2)
        preds = torch.cat([pred_origin, pred_target],dim=0)
        preds_v2 = torch.cat([pred_origin_v2, pred_target_v2],dim=0)
        assert(labels.shape[0]==preds.shape[0])
        gan_loss = nn.CrossEntropyLoss()(preds, labels)
        gan_loss_v2 = nn.CrossEntropyLoss()(preds_v2, labels)
        return gan_loss, gan_loss_v2

    def loss(self, samples, train=True):
        """
        Computes the Distributional Q-learning loss, based on projecting the
        discounted rewards + target Q-distribution into the current Q-domain,
        with cross-entropy loss.

        Returns loss and KL-divergence-errors for use in prioritization.
        """
        #:param: samples (t,b,..)
        # samples.observation (t,b,c,h,w) float
        # samples.action (t,b,1) or (t,b) long
        # samples.reward (t,b,1) or (t,b) float
        # samples.done (t,b,1) or (t,b) bool
        
        pred_qs, pred_rew, spr_loss, pred_reward_real, reconstruction_loss, pred_obss, \
            pred_qs_fake, pred_rew_fake= self.model(samples.observation,
                         samples.action,
                         samples.reward)  # [B,A,P]
        
        self.log_twin_sequence((pred_obss[:,0]*255).to(torch.uint8),samples.observation[:self.horizon+1,0],train=train)
        #random record obs
        
        #rl_loss
        #pred_qs: list of (bz, act_dim)
        
        rl_loss = self.dqn_rl_loss(pred_qs[0], samples, 0)
        rl_loss_fake = self.dqn_rl_loss(pred_qs_fake[0], samples, 0)

        #reward_loss
        if len(pred_rew) > 0:
            pred_rew = torch.stack(pred_rew, 0)
            pred_rew_fake = torch.stack(pred_rew_fake, 0)
            #(t,bz,2*limit+1)
            with torch.no_grad():
                reward_target = to_categorical(samples.reward[:self.horizon+1].flatten(), limit=1).view(*pred_rew.shape)
            reward_loss = -torch.sum(reward_target * pred_rew, 2).mean(0)
            reward_loss_fake = -torch.sum(reward_target * pred_rew_fake, 2).mean(0)
        else:
            reward_loss = torch.zeros(samples.observation.shape[1],)
            reward_loss_fake = torch.zeros(samples.observation.shape[1],)
        model_rl_loss = torch.zeros_like(reward_loss)

        '''
        if self.model_rl_weight > 0:
            #rl loss for hidden steps
            for i in range(1, self.horizon+1):
                    jump_rl_loss, model_KL = self.rl_loss(log_pred_ps[i],
                                                   samples,
                                                   i)
                    model_rl_loss = model_rl_loss + jump_rl_loss
        '''
        #if terminal, truncate spr_loss
        nonterminals = 1. - torch.sign(torch.cumsum(samples.done.squeeze(), 0)).float()
        nonterminals = nonterminals[:self.horizon + 1]
        spr_loss = spr_loss*nonterminals
        
        #why seperate spr loss?
        if self.horizon > 0:
            model_spr_loss = spr_loss[1:].mean(0)
            spr_loss = spr_loss[0]
        else:
            spr_loss = spr_loss[0]
            model_spr_loss = torch.zeros_like(spr_loss)

        spr_loss = spr_loss
        model_spr_loss = model_spr_loss
        reward_loss = reward_loss

        pred_qs =  torch.stack(pred_qs, 0)
        pred_reward_real =  torch.stack(pred_reward_real, 0)

        gan_loss, gan_loss_v2 = self.compute_gan_loss(pred_obss, samples.observation[:self.horizon+1])

        return rl_loss.mean(), \
               model_rl_loss.mean(),\
               reward_loss.mean(), \
               spr_loss.mean(), \
               model_spr_loss.mean(), pred_qs.mean(), pred_reward_real.mean(), \
                reconstruction_loss, gan_loss, gan_loss_v2, rl_loss_fake.mean(), reward_loss_fake.mean()

    def optimize_agent(self, sample):
        self.iter += 1
        rl_loss, model_rl_loss, reward_loss, spr_loss, model_spr_loss, pred_qs_mean, \
        pred_rew_mean, reconstruction_loss, gan_loss, gan_loss_v2,rl_loss_fake, reward_loss_fake = self.loss(sample)
        reconstruction_loss = reconstruction_loss*10*max((1+self.iter/3000),5)
        #5000step左右就能把整体框架学出来
        #这里最好搞成linear的系数
        rl_loss = rl_loss*5
        #
        with torch.autograd.set_detect_anomaly(True):
            total_loss = spr_loss + model_spr_loss + \
            reconstruction_loss# - gan_loss
            # rl_loss reward_loss
            #discriminator
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # self.gan_optimizer.zero_grad()
            # gan_loss_v2.backward()
            # self.gan_optimizer.step()
        
        #log
        self.writer.add_scalar('Loss/rl_loss', rl_loss.detach().cpu(), self.iter)
        self.writer.add_scalar('Loss/model_rl_loss', model_rl_loss.detach().cpu(), self.iter)
        self.writer.add_scalar('Loss/reward_loss', reward_loss.detach().cpu(), self.iter)
        self.writer.add_scalar('Loss/spr_loss', spr_loss.detach().cpu(), self.iter)
        self.writer.add_scalar('Loss/model_spr_loss', model_spr_loss.detach().cpu(), self.iter)
        self.writer.add_scalar('Loss/total_loss', total_loss.detach().cpu(), self.iter)
        self.writer.add_scalar('Loss/reconstruction_loss', reconstruction_loss.detach().cpu(), self.iter)
        self.writer.add_scalar('Loss/gan_loss', gan_loss.detach().cpu(), self.iter)
        self.writer.add_scalar('Model/pred_qs', pred_qs_mean.detach().cpu(), self.iter)
        self.writer.add_scalar('Model/pred_rew', pred_rew_mean.detach().cpu(), self.iter)
        self.writer.add_scalar('Data/reward', sample.reward.mean(), self.iter)

        self.fake_writer.add_scalar('Loss/rl_loss',rl_loss_fake.detach().cpu(), self.iter)
        self.fake_writer.add_scalar('Loss/reward_loss',reward_loss_fake.detach().cpu(), self.iter)
        return total_loss
        
    def validation_step(self):
        batch = next(self.valid_loaders)
        sample = self.transform_sample(batch)
        #generate_sample
        rl_loss, model_rl_loss, reward_loss, spr_loss, model_spr_loss, pred_qs_mean, \
        pred_rew_mean, reconstruction_loss, gan_loss, gan_loss_v2,_,_ = self.loss(sample, train=False)
        reconstruction_loss = reconstruction_loss*10*max((1+self.iter/3000),5)
        #5000step左右就能把整体框架学出来
        rl_loss = rl_loss*5

        with torch.autograd.set_detect_anomaly(True):
            total_loss = spr_loss + model_spr_loss + \
             reconstruction_loss#rl_loss + reward_loss +
        
        #log
        self.valid_writer.add_scalar('Loss/rl_loss', rl_loss.detach().cpu(), self.iter)
        self.valid_writer.add_scalar('Loss/model_rl_loss', model_rl_loss.detach().cpu(), self.iter)
        self.valid_writer.add_scalar('Loss/reward_loss', reward_loss.detach().cpu(), self.iter)
        self.valid_writer.add_scalar('Loss/spr_loss', spr_loss.detach().cpu(), self.iter)
        self.valid_writer.add_scalar('Loss/model_spr_loss', model_spr_loss.detach().cpu(), self.iter)
        self.valid_writer.add_scalar('Loss/total_loss', total_loss.detach().cpu(), self.iter)
        self.valid_writer.add_scalar('Loss/reconstruction_loss', \
        reconstruction_loss.detach().cpu(), self.iter)
        self.valid_writer.add_scalar('Data/reward', sample.reward.mean(), self.iter)

    def transform_sample(self,batch):
        observation, action, reward, done = batch
        observation, action, reward, done = [item.view(64,16,*item.shape[1:]).to(self.device).permute(1,0,\
        *tuple(range(2,len(item.shape)+1)))
        for item in batch]
        sample = Samples(observation = observation, action = action.to(torch.long), 
        reward = reward, done = done)
        return sample

    def train(self, n_epoch):
        for _ in range(n_epoch):
            print(f'new epoch, iter {self.iter}')
            counter = 0
            start_time = time.time()
            for loader in self.train_loaders:
                print('new loader')
                for batch in loader:
                    sample = self.transform_sample(batch)
                    self.optimize_agent(sample)

                    counter+=1

                    if counter>100:
                        end_time = time.time()
                        self.writer.add_scalar('Training/speed', end_time-start_time, self.iter)
                        start_time = end_time
                        counter = 0
                        #validation_step
                        self.validation_step()
                        #validation
            


if __name__=='__main__':
    pass
    # algo = DQNSPR()
    # samples = Samples(observation = torch.randn((8,10,4,84,84)),
    # action = torch.randint(0,5,(8,10)), reward = torch.randn((8,10)), 
    # done = torch.randint(0,2,(8,10)))#(t,b,...)
    #print(algo.loss(samples))

