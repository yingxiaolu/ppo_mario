from icecream import ic
import sys
import torch
import gym_super_mario_bros
from matplotlib import pyplot as plt
from ppo_model import PPO
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device='cpu'
import torch.nn.functional as F
from customenv import Customenv
from torch.distributions import Categorical,MultivariateNormal
env=Customenv()
import os
import numpy as np
import fire
import copy
import time

# Model Param
CHECK_FREQ_NUMB = 10000 #每训练多少帧保存一次模型
TOTAL_TIMESTEP_NUMB = 500000 #总训练帧数
N_STEPS = 512 # Number of steps to collect to train on
EPISODE_NUMBERS = 20
MAX_TIMESTEP_TEST = 1000


LEARNING_RATE = 0.00001 #学习率
# GAE = 1.0   # 控制优势估计的偏差和方差
ENT_COEF = 0.01 # Entropy Coefficient
GAMMA = 0.9#单帧奖励计算折扣
# LAMBDA = 0.95
# TAU=1.0 #控制GAE的偏差和方差
EPSILON = 0.1 # 裁剪范围
BATCH_SIZE = 600 #一个batch内游戏次数
EPOCHS = 10000 # Number of Epochs
N_EPOCHS=10 #训练完一个batch后再迭代跟新的次数
WARM_UP=1 #在训练初期, 非常容易死, 导致单次帧很少, 此时加大batch_size.

ic() 

torch.manual_seed(0)
model_path='./model.pth'
model=torch.load(model_path)
# model=PPO(1,5).to(device)
    
def test():
    state = env.custom_reset()
    env.render=True
    reward_list=[]
    
    state = env.custom_reset()
    info= env.info
    done=False
    while not done:
        # state = torch.from_numpy(np.concatenate(state, 0)).to(device).float()
        # ic(state.shape)
        state=state[:,:,:,-1].copy()
        # ic(state.shape)
        # state=state.transpose(2,0,1)
        # ic(state.shape)
        state=np.expand_dims(state,axis=0)
        
        logits,_=model(torch.FloatTensor(state).to(device))
        # print(logits)
        # batch_vals.append(value.detach())
        policy=F.softmax(logits.detach(),dim=1)
        #得到policy中,值最大的下标
        action=torch.argmax(policy)
        state,reward,done,info=env.custom_step(action.item())
        ic(reward,info,action)
        reward_list.append(reward)
        time.sleep(0.1)
    print(reward_list)
test()