from src.models import *
from src.utils import set_config, gen_args
from src.algo import *
from dopamine.replay_memory import circular_replay_buffer
import cv2
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os 

os.environ["CUDA_VISIBLE_DEVICES"]='3'
class ReplayBuffer(Dataset):
    def __init__(self, path = '/home/lzy/Breakout/1/replay_logs', suffix='50'):
        super().__init__()
        self.buffer = circular_replay_buffer.OutOfGraphReplayBuffer((4,84,84),4,100000,32)
        self.buffer.load(path,suffix=suffix)
        self.length = (self.buffer._store['reward'].shape[0] - 30)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        #stack observation
        obs_stacked = np.stack([self.buffer._store['observation'][index-3+i] for i in range(4)],axis=0)
        return obs_stacked, self.buffer._store['action'][index], \
        self.buffer._store['reward'][index], self.buffer._store['terminal'][index]

def gen_train_loader_list():
    loader_list = []
    for index in range(1,5):
        for i in range(3):
            dataset = ReplayBuffer(path = f'/home/lzy/Breakout/{index}/replay_logs', suffix=str(i*20))
            print(len(dataset))
            dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=True, num_workers=4)
            loader_list.append(dataloader)
    return loader_list

def gen_valid_loader_list():
    loader_list = []
    for i in range(3):
        dataset = ReplayBuffer(path = '/home/lzy/Breakout/5/replay_logs', suffix=str(i*20))
        print(len(dataset))
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=True, num_workers=4)
        loader_list.append(dataloader)
    return loader_list

train_loaders = gen_train_loader_list()
valid_loaders = gen_valid_loader_list()
algo = DQNSPR(train_loaders, valid_loaders)
print('begin training')
algo.train(20)


