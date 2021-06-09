
import torch
import torch.nn.functional as F
import torch.nn as nn

from rlpyt.models.utils import scale_grad, update_state_dict
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from src.utils import count_parameters, dummy_context_mgr
import numpy as np
from kornia.augmentation import RandomAffine,\
    RandomCrop,\
    CenterCrop, \
    RandomResizedCrop
from kornia.filters import GaussianBlur2d
import copy

class SPRDqnModel(torch.nn.Module):
    def __init__(self, horizon):   
        super().__init__()
        #data augmentation
        self.define_augmentation()
        self.momentum_tau = 0.01
        self.horizon = horizon
        self.in_channels = in_channels = 4
        imagesize = 84
        output_size = 4
        self.dqn_hidden_size = 512
        self.renormalize = True
        self.aug_prob=0
        self.device='cuda'
        #encoder
        self.encoder = Conv2dModel(
            in_channels=in_channels,
            channels=[64, 128, 256],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=[0, 0, 0],
            use_maxpool=False,
            dropout=0,
        )
        fake_input = torch.zeros(1, in_channels, imagesize, imagesize)
        fake_output = self.encoder(fake_input)
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1]*fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))
        self.num_actions = output_size

        #RL_head
        self.head = DQNHeadModel(self.hidden_size,
                            output_size,
                            hidden_size=self.dqn_hidden_size,
                            pixels=self.pixels)

        #prediction
        self.dynamics_model = TransitionModel(channels=self.hidden_size,
                                                num_actions=output_size,
                                                pixels=self.pixels,
                                                hidden_size=self.hidden_size,
                                                limit=1)

        
        self.define_spr()

        print("Initialized model with {} parameters".format(count_parameters(self)))

    def define_spr(self):
        self.local_classifier = nn.Sequential(nn.Flatten(-3, -1),
                                                nn.Linear(self.hidden_size*self.pixels,self.hidden_size),
                                                nn.BatchNorm1d(self.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size,
                                                        self.hidden_size))
        self.local_final_classifier = nn.Identity()

        self.target_encoder = Conv2dModel(
            in_channels=self.in_channels,
            channels=[64, 128, 256],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=[0, 0, 0],
            use_maxpool=False,
            dropout=0,
        )
        self.local_target_classifier = nn.Sequential(nn.Flatten(-3, -1),
                                                nn.Linear(self.hidden_size*self.pixels,self.hidden_size),
                                                nn.BatchNorm1d(self.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size,
                                                        self.hidden_size))
        for param in (list(self.target_encoder.parameters())
            + list(self.local_target_classifier.parameters())):
            param.requires_grad = False

    def define_augmentation(self):
        self.transforms = []
        self.eval_transforms = []
        augmentation = ['none']
        self.uses_augmentation = False
        for aug in augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "crop":
                transformation = RandomCrop((84, 84))
                # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
                eval_transformation = CenterCrop((84, 84))
                self.uses_augmentation = True
                imagesize = 84
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
                eval_transformation = nn.Identity()
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                #maybe transform?
                image = maybe_transform(image, transform,
                                        eval_transform, p=self.aug_prob)
        return image

    def latent_spr_loss(self, latents, target_latents)->torch.Tensor:
        #:param: latents: (t*b, 64,7,7)
        #:param: target_latents: (t*b, 64,7,7)
        #:return: loss: ()
        projected_latents = self.local_classifier(latents)
        projected_latents = self.local_final_classifier(projected_latents)
        with torch.no_grad():
            projected_targets = self.local_target_classifier(target_latents)

        projected_latents = projected_latents.view(-1,self.hidden_size)
        projected_targets = projected_targets.view(-1,self.hidden_size)
        f_x1 = F.normalize(projected_latents.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(projected_targets.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1)
        return loss


    def do_spr_loss(self, pred_latents, observation):
        #:param: latents: (length,bz, 64,7,7)
        #:param: obervation: (length,bz, c,84,84)
        #:return: spr_loss: (length,bz)
        #current latents and future observations
        length, bz = pred_latents.shape[:2]
        #flatten_dim
        pred_latents = pred_latents.contiguous().view(bz*length, *pred_latents.shape[2:])
        #observation(6,2,c,84,84)
        observation =  observation.contiguous().view(bz*length, *observation.shape[2:])
        #data aug
        target_images = self.transform(observation, False)
        #flat here
        
        with torch.no_grad():
            target_latents = self.target_encoder(target_images)

        spr_loss = self.latent_spr_loss(pred_latents, target_latents)
        spr_loss = spr_loss.view(length, bz)
        #update target   
        update_state_dict(self.target_encoder,self.encoder.state_dict(),
            self.momentum_tau)
        update_state_dict(self.local_target_classifier,
                            self.local_classifier.state_dict(),
                            self.momentum_tau)
        
        return spr_loss

    def stem_forward(self, img):
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # :param: img: (B,C,84,84)
        # :return: (B,64,7,7)
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.encoder(img.view(T * B, *img_shape))  # Fold if T dimension.
        if self.renormalize:
            conv_out = renormalize(conv_out, -3)
        return conv_out

    def head_forward(self,
                     conv_out):
        #:param: conv_out: (B,64,7,7)
        #:return: (B,action_dim)
        lead_dim, T, B, img_shape = infer_leading_dims(conv_out, 3)
        p = self.head(conv_out)

        p = p.squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p

    def forward(self, observation,
                prev_action, prev_reward):
        #:param: observation: (t,b,c,h,w)
        #:param: prev_action: (t,b,1)
        #:param: prev_reward: (t,b,1)
        #:return: pred_qs: list of (bz, act_dim)
        #:return: pred_reward list of (bz,rewdim)
        #:return: spr_loss

        # cut observation to horizon
        
        if observation.shape[0] > self.horizon:
            observation = observation[:self.horizon+1]#(t,b,c,84,84)
        length, batch_size = self.horizon+1, observation.shape[1]
        pred_qs = []
        pred_reward = []
        pred_latents = []
        pred_reward_real = []
        pred_obss = []
        reward_dist = torch.Tensor([-1,0,1]).to(self.device)
        input_obs = observation[0]
        #FIXME: add augmentation
        input_obs = self.transform(input_obs, augment=False)
        
        latent = self.stem_forward(input_obs)
        #latent: (b,64,7,7)
        pred_qs.append(self.head_forward(latent))
        pred_latents.append(latent)
        
        if self.horizon > 0:
            pred_rew = self.dynamics_model.reward_predictor(pred_latents[0])
            pred_reward.append(F.log_softmax(pred_rew, -1))
            real_reward = F.softmax(pred_rew,-1)*(\
            reward_dist.repeat(pred_rew.shape[0],1))
            #(b)
            pred_reward_real.append(real_reward.mean(-1))

            pred_obs =  self.dynamics_model.decoder(pred_latents[0])
            pred_obss.append(pred_obs)

            for j in range(1, self.horizon + 1):
                latent, pred_rew, pred_obs, current_obs = self.step(latent, prev_action[j])
                #pred_rew: (batch, 3)
                pred_rew = pred_rew[:observation.shape[1]]
                pred_latents.append(latent)
                pred_reward.append(F.log_softmax(pred_rew, -1))
                real_reward = F.softmax(pred_rew,-1)*(\
                reward_dist.repeat(pred_rew.shape[0],1))
                pred_reward_real.append(real_reward.mean(-1))

                pred_obss.append(pred_obs)


        #predict Q values
        #pred_latents[i]: (b,64,7,7)
        #prev_action[i]: (b,1)
        #prev_reward[i]: (b,reward_dim)
        for i in range(1, len(pred_latents)):
            pred_qs.append(self.head_forward(pred_latents[i]))
        
        pred_qs_fake = [torch.rand(pred_qs[0].shape,device=self.device)+1 \
         for i in range(len(pred_latents))]
        pred_rew_fake = [torch.Tensor([0.01,0.98,0.01]).log().repeat(batch_size,1).to(self.device) \
          for i in range(len(pred_reward))]
        #here
        pred_latents = torch.cat([ p.unsqueeze(0) for p  in pred_latents], dim=0)
        #(length, bz, 64, 7, 7)
        spr_loss = self.do_spr_loss(pred_latents, observation)

        pred_obss = torch.cat([p.unsqueeze(0) for p  in pred_obss], dim=0)

        target_obs = self.transform(observation, False)

        reconstruction_loss = torch.square(pred_obss- target_obs).mean()
        

        return pred_qs, pred_reward, spr_loss, pred_reward_real, \
        reconstruction_loss, pred_obss, pred_qs_fake, pred_rew_fake

    def step(self, state, action):
        next_state, reward_logits, next_obs, current_obs = self.dynamics_model(state, action)
        return next_state, reward_logits, next_obs, current_obs
    
    def target_q(self, img):
        #:param: img: (b,c,84,84)
        #:return: (b,act_dim)
        img = self.transform(img, augment=False)
        conv_out = self.stem_forward(img)
        return self.head_forward(conv_out)


    @torch.no_grad()
    def transform(self, images: torch.Tensor, augment=False):
        augment=False
        #:param: images: (B,C,H,W)
        #:return: the same shape
        images = images.float()/255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        #the same shape
        if augment:
            processed_images = self.apply_transforms(self.transforms,
                                                     self.eval_transforms,
                                                     flat_images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms,
                                                     None,
                                                     flat_images)
        
        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

def maybe_transform(image, transform, alt_transform, p=0.8):
    #random transform to some extent
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        base_images = alt_transform(image)
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * base_images
        return processed_images

class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
            dropout=0.,
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        #:param :input (b,c,84,84)
        #:return: (b,64,7,7)
        return self.conv(input)



class TransitionModel(nn.Module):
    #what's input and output?
    def __init__(self,
                 channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 pixels=36,
                 limit=300,
                 action_dim=6,
                 norm_type="bn",
                 renormalize=True,
                 residual=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.renormalize = renormalize
        self.residual = residual
        layers = [Conv2dSame(channels+num_actions, hidden_size, 3),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size,
                                        hidden_size,
                                        norm_type))
        layers.extend([Conv2dSame(hidden_size, channels, 3)])

        self.action_embedding = nn.Embedding(num_actions, pixels*action_dim)
        #(6->7*7*6)
        self.network = nn.Sequential(*layers)
        #2 conv,70 channels to 64 channels
        self.reward_predictor = RewardPredictor(channels,
                                                pixels=pixels,
                                                limit=limit,
                                                norm_type=norm_type)
        self.decoder = Decoder()
        self.train()

    def forward(self, x, action):
        #:param: x:(32,64,7,7)
        #:param: action:(32)
        #:return: next_state:(32,64,7,7)
        #:return: next_reward:(32,3)
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0],
                                    self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        action_onehot[batch_range, action, :, :] = 1
        stacked_image = torch.cat([x, action_onehot], 1)
        next_state = self.network(stacked_image)
        if self.residual:
            next_state = next_state + x
        next_state = F.relu(next_state)
        if self.renormalize:
            next_state = renormalize(next_state, 1)
        next_reward = self.reward_predictor(next_state)
        next_obs = self.decoder(next_state)
        current_obs = self.decoder(x)
        return next_state, next_reward, next_obs, current_obs


class RewardPredictor(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_size=1,
                 pixels=36,
                 limit=300,
                 norm_type="bn"):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 256),
                  nn.ReLU(),
                  nn.Linear(256, limit*2 + 1)]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):

        return self.network(x)


def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "none", None]
    if type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=affine)
        else:
            return nn.BatchNorm2d(channels, affine=affine)
    elif type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return nn.GroupNorm(1, channels, affine=affine)
    elif type == "in":
        return nn.GroupNorm(channels, channels, affine=affine)
    elif type == "none" or type is None:
        return nn.Identity()

class Conv2dSame(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                            stride=stride, padding=ka)
        )

    def forward(self, x):
        return self.net(x)


class DQNHeadModel(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=256,
                 pixels=30):
        super().__init__()
        linear = nn.Linear
        self.linears = [linear(input_channels*pixels, hidden_size),
                        linear(hidden_size, output_size)]
        layers = [nn.Flatten(-3, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.network = nn.Sequential(*layers)
        self.network.apply(weights_init)
        self._output_size = output_size

    def forward(self, input):
        return self.network(input).view(-1, self._output_size)

def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type="bn"):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            init_normalization(out_channels, norm_type),
            Conv2dSame(out_channels, out_channels, 3),
            init_normalization(out_channels, norm_type),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self,
                 norm_type="in"):
        super().__init__()
        #transpose conv
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 256, out_channels =128 , kernel_size = 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 128, out_channels =64 , kernel_size = 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 64, out_channels =32 , kernel_size = 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels =16 , kernel_size = 5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 16, out_channels =4 , kernel_size = 4, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        #:param: (t*b,64,7,7)
        #:return: (t*b, c,84,84)
        return (self.block(x)[:,:,4:88,4:88]+1)/2

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        #transpose conv
        self.encoder = Conv2dModel(
            in_channels=4,
            channels=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=[0, 0, 0],
            use_maxpool=False,
            dropout=0,
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*64, 128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self, x):
        #:param: (t*b,c,84,84)
        #:return: (t*b,2)
        x = self.encoder(x)
        return self.head(x)


def weights_init(m):
    if isinstance(m, Conv2dSame):
        torch.nn.init.kaiming_uniform_(m.layer.weight, nonlinearity='linear')
        torch.nn.init.zeros_(m.layer.bias)
    elif isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
        torch.nn.init.zeros_(m.bias)

def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit*2+1), device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return distribution

if __name__=='__main__':
    # model = SPRDqnModel()
    # fake_input= torch.randn((2,4,84,84))
    model = Decoder()
    fake_latent = torch.randn((2,256,7,7))
    print(model(fake_latent).shape)
    # pred_qs, pred_reward, spr_loss = (model.forward(torch.randn(6,2,4,84,84), torch.randint(0,5,(6,2,1)),torch.randn(6,2,1)))
    # print(pred_qs[0].shape)
    # print(pred_reward[0].shape)
    # print(spr_loss.shape)

