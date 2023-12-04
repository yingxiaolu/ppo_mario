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
EPOCHS = 10000 # Number of Epochs
if sys.platform.startswith('linux'):
    N_EPOCHS= 50
    BATCH_TIMESTEP_NUMB = 8000 #
else:
    N_EPOCHS=20
    BATCH_TIMESTEP_NUMB = 12000 #

ic() 

# actor_saved_path='./actor.pth'
# critic_saved_path='./critic.pth'

# if os.path.exists(critic_saved_path):
#     actor=torch.load(actor_saved_path)
#     critic=torch.load(critic_saved_path)
# else:
#     actor=PPO(1,7).to(device)
#     critic=PPO(1,7).to(device)
torch.manual_seed(1223)
model_path='./model.pth'
# if os.path.exists(model_path):
#     model=torch.load(model_path)
# else:
model=PPO(1,5).to(device)
# model=PPO(1,7).to(device)
# for name, param in model.named_parameters():
#     print(f"Parameter name: {name}, Device: {param.device}")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)
# cov_var = torch.full(size=(3,), fill_value=0.5).to(device)
# cov_mat = torch.diag(cov_var).to(device)

final_x_pos_list=[]
timesteps_list=[]
avge_final_xpos_in_one_batch=[]
avge_reward_in_one_batch=[]
def get_batchs_data(env,batch_timestep_numb):
    '''
    一次从头到尾跑完一趟游戏, 多趟组成一个batchs
    '''
    global final_x_pos_list,timesteps_list,avge_final_xpos_in_one_batch,avge_reward_in_one_batch
    batch_obs, batch_acts, batch_log_probs,batch_rews, rewards, batch_lens,batch_vals,dones=[],[],[],[],[],[],[],[]
    final_state=None
    batch_final_x_pos=[]
    # barch_rew=[]
    cur_timesteps=0
    while True:
        state = env.custom_reset()
        info= env.info
        done=False
        timesteps=[]
        ep_rews=[]

        while not done:
            state=state[:,:,:,-1].copy()
            state=np.expand_dims(state,axis=0)
            state=torch.FloatTensor(state).to(device)# torch.Size([1, 1, 84, 84])
            # ic(state.shape)
            batch_obs.append(state)
            # with torch.no_grad():
            logits,value=model(state)
            # print(logits)
            batch_vals.append(value)
            policy=F.softmax(logits,dim=1)#.detach()
            dist = Categorical(policy)
            action=dist.sample()
            log_prob=dist.log_prob(action)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)
            state,reward,done,info=env.custom_step(action.item())
            # ic(action.item())
            # ic(reward)
            dones.append(done)
            rewards.append(reward)
            ep_rews.append(reward)
            timesteps.append(1)
            cur_timesteps+=1
            
            if cur_timesteps>=batch_timestep_numb:
                break
            
        batch_final_x_pos.append(info['x_pos'])
        # ic(len(timesteps))
        timesteps_list.append(len(timesteps))
        batch_lens.append(len(timesteps))
        batch_rews.append(ep_rews)
        avge_reward_in_one_batch.append(np.mean(ep_rews))
        
        if cur_timesteps>=batch_timestep_numb:
            break
        
    final_x_pos_list+=batch_final_x_pos
    avge_final_xpos_in_one_batch.append(np.mean(batch_final_x_pos))
    batch_rtgs=[]
    for rews in batch_rews[::-1]:
        discounted_reward=0
        for rew in rews[::-1]:
            discounted_reward=rew+GAMMA*discounted_reward
            batch_rtgs.insert(0,discounted_reward)
            
    batch_obs=torch.cat(batch_obs,dim=0)
    batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(device)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(device)
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(device)
    batch_vals=torch.tensor(batch_vals,dtype=torch.float).to(device)
    
    ic(avge_final_xpos_in_one_batch[-1])
    plt.figure(figsize=(8*8,8*2))
    plt.subplot(4,1,1)
    plt.plot(final_x_pos_list,color='red')
    plt.legend(['final_x_pos'])
    plt.subplot(4,1,2)
    plt.plot(timesteps_list,color='blue')
    plt.legend(['timesteps'])
    plt.subplot(4,1,3)
    plt.plot(avge_final_xpos_in_one_batch,color='green')
    plt.legend(['avge_xpos_in_one_batch'])
    plt.subplot(4,1,4)
    plt.plot(avge_reward_in_one_batch,color='black')
    plt.legend(['avge_reward_in_one_batch'])
    plt.savefig('./info.png')
    plt.close()
    
    return batch_obs, batch_acts, batch_log_probs,batch_rtgs, rewards, batch_lens,batch_vals,dones
      
def train(batch_timestep_numb=BATCH_TIMESTEP_NUMB):
    state = env.custom_reset()
    for epoch in range(EPOCHS):
        print(f'epoch:{epoch}')
        batch_obs, batch_acts, batch_log_probs,batch_rtgs, rewards, batch_lens,batch_vals,dones=get_batchs_data(env,batch_timestep_numb)
        # ic(batch_obs.shape,batch_acts.shape,batch_log_probs.shape,batch_rtgs.shape,batch_vals.shape,len(rewards),len(batch_lens),len(dones))
        # l,r=100,120
        # print(batch_acts[l:r])
        # print(batch_log_probs[l:r])
        # print(batch_rtgs[l:r])
        # print(batch_vals[l:r])
        # print(rewards[l:r])
        
        A_k=batch_rtgs-batch_vals.detach()
        # print(A_k[l:r])
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        # print(A_k[l:r])
        # sys.exit()
        for nepoch in range(N_EPOCHS):
            logits, value = model(batch_obs)
            
            policy = F.softmax(logits, dim=1)
            m = Categorical(policy)
            log_probs = m.log_prob(batch_acts)
            
            # dist=MultivariateNormal(logits,cov_mat)
            # log_probs=dist.log_prob(batch_acts)
            
            ratio = torch.exp(log_probs - batch_log_probs)
            # ic(policy[:20])
            # ic(new_log_probs[:20])
            # ic(logits.shape,policy.shape,new_log_probs.shape,ratio.shape)
            surr1 = ratio * A_k
            surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * A_k
            entropy_loss = (m.entropy()).mean()
            
            actor_loss = -(torch.min(surr1, surr2)).mean()
            # critic_loss = F.smooth_l1_loss(reward+GAMMA*next_values, value.squeeze())
            critic_loss = F.smooth_l1_loss(batch_vals, value.squeeze())
            
            loss = critic_loss + actor_loss - ENT_COEF * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'actor_loss:{actor_loss.item()},critic_loss:{critic_loss.item()},entropy_loss:{entropy_loss.item()}')
        # if epoch % 10 == 0:
        torch.save(model,model_path)
        

if __name__ == '__main__':
    # fire.Fire()
    train()















        # # GAE=[]
        # # next_values=[]
        # # for value, reward, done,ob in list(zip(batch_vals, rewards, dones,batch_obs))[::-1]:
        # #     if done:
        # #         gae = 0
        # #         ob=ob.reshape(1,1,84,84)
        # #         _,next_value=model(ob)
        # #         next_value=next_value.detach()
        # #     next_values.append(next_value)
        # #     next_value=value
        # # next_values=torch.tensor(next_values[::-1],dtype=torch.float).to(device)
        # # GAE=get_gaes(rewards, batch_vals, next_values, GAMMA, LAMBDA)[0]
        # # GAE=torch.tensor(GAE,dtype=torch.float).to(device).reshape(-1)
        # #     GAE.append(gae)
        # # GAE = GAE[::-1]
        # # GAE = torch.tensor(GAE, dtype=torch.float).to(device).reshape(-1)
  
        # _, next_value, = model(batch_obs[-1,:,:,:].reshape(1,1,84,84))
        # next_value = next_value.squeeze()
        # gae = 0
        # R = []
        # for value, reward, done in list(zip(batch_vals, rewards, dones))[::-1]:
        #     gae = gae * GAMMA * LAMBDA *(1-done)
        #     # ic(value.device)
        #     gae = gae + reward + GAMMA * next_value.item() * (1 - done) - value.item()
            
        #     next_value = value
        #     R.append(gae + value.item())
        # R = R[::-1]
        # R = torch.tensor(R, dtype=torch.float).to(device).reshape(-1)
        # advantages = R - batch_vals
  
        # # ic(GAE.shape)
        # # ic(batch_vals.shape)
        # # ic(len(rewards))
        # # ic(next_values.shape)
        # # ic(GAE[:30])#很大
        # # ic(batch_vals[:10])
        # # ic(rewards[:10])
        # # ic(next_values[:10])
        # # ic(batch_vals[:10])