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
from torch.distributions import Categorical
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


LEARNING_RATE = 0.0001 #学习率
GAE = 1.0   # 控制优势估计的偏差和方差
ENT_COEF = 0.01 # Entropy Coefficient
GAMMA = 0.99#单帧奖励计算折扣
LAMBDA = 0.95
# TAU=1.0 #控制GAE的偏差和方差
EPSILON = 0.1 # 裁剪范围
BATCH_SIZE = 200 #一个batch内游戏次数
EPOCHS = 10000 # Number of Epochs
N_EPOCHS=5 #训练完一个batch后再迭代跟新的次数
WARM_UP=1 #在训练初期, 非常容易死, 导致单次帧很少, 此时加大batch_size.

ic() 

# actor_saved_path='./actor.pth'
# critic_saved_path='./critic.pth'

# if os.path.exists(critic_saved_path):
#     actor=torch.load(actor_saved_path)
#     critic=torch.load(critic_saved_path)
# else:
#     actor=PPO(1,7).to(device)
#     critic=PPO(1,7).to(device)
torch.manual_seed(0)
model_path='./model.pth'
model=PPO(1,3).to(device)
# model=PPO(1,7).to(device)
# for name, param in model.named_parameters():
#     print(f"Parameter name: {name}, Device: {param.device}")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

final_x_pos_list=[]
timesteps_list=[]
final_reward_list=[]
def get_batchs_data(env,batch_size=BATCH_SIZE):
    '''
    一次从头到尾跑完一趟游戏, 多趟组成一个batchs
    '''
    batch_obs, batch_acts, batch_log_probs,batch_rews, rewards, batch_lens,batch_vals,dones=[],[],[],[],[],[],[],[]
    final_state=None
    for i in range(batch_size):
        state = env.custom_reset()
        info= env.info
        done=False
        timesteps=[]

        while not done:
            state=state[:,:,:,-1].copy()
            state=np.expand_dims(state,axis=0)
            state=torch.FloatTensor(state).to(device)# torch.Size([1, 1, 84, 84])
            # ic(state.shape)
            batch_obs.append(state)
            # with torch.no_grad():
            logits,value=model(state)
            # print(logits)
            batch_vals.append(value.detach())
            policy=F.softmax(logits.detach(),dim=1)
            # print(policy)
            old_m = Categorical(policy)
            action=old_m.sample()
            # print(action)
            batch_acts.append(action)
            batch_log_probs.append(old_m.log_prob(action))
            state,reward,done,info=env.custom_step(action.item())
            # ic(action.item())
            # ic(reward)
            dones.append(done)
            rewards.append(reward)
            timesteps.append(1)
        # ic(len(timesteps))
        final_x_pos_list.append(info['x_pos'])
        timesteps_list.append(len(timesteps))
        batch_lens.append(len(timesteps))
        

    

    batch_obs=torch.cat(batch_obs,dim=0)
    batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(device)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(device)
    # batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(device)
    batch_vals=torch.tensor(batch_vals,dtype=torch.float).to(device)
    
    plt.figure(figsize=(8*8,8*2))
    #做上下两个子图
    plt.subplot(2,1,1)
    plt.plot(final_x_pos_list,color='red')
    plt.legend(['final_x_pos'])
    plt.subplot(2,1,2)
    plt.plot(timesteps_list,color='blue')
    plt.legend(['timesteps'])
    plt.savefig('./info.png')
    plt.close()
    
    return batch_obs, batch_acts, batch_log_probs, rewards, batch_lens,batch_vals,dones

def get_gaes(rewards, state_values, next_state_values, GAMMA, LAMBDA):
    deltas = [r_t + GAMMA * next_v - v for r_t, next_v, v in zip(rewards, next_state_values, state_values)]
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):
        gaes[t] = gaes[t] + LAMBDA * GAMMA * gaes[t + 1]
    return gaes, deltas
      
def train():
    state = env.custom_reset()
    warm_up=WARM_UP
    for epoch in range(EPOCHS):
        print(f'epoch:{epoch}')
        batch_size=BATCH_SIZE
        # if warm_up>=1:
        #     batch_size=BATCH_SIZE*warm_up
        #     warm_up-=1
        batch_obs, batch_acts, batch_log_probs, rewards, batch_lens,batch_vals,dones=get_batchs_data(env,batch_size=batch_size)
        # GAE=[]
        # next_values=[]
        # for value, reward, done,ob in list(zip(batch_vals, rewards, dones,batch_obs))[::-1]:
        #     if done:
        #         gae = 0
        #         ob=ob.reshape(1,1,84,84)
        #         _,next_value=model(ob)
        #         next_value=next_value.detach()
        #     next_values.append(next_value)
        #     next_value=value
        # next_values=torch.tensor(next_values[::-1],dtype=torch.float).to(device)
        # GAE=get_gaes(rewards, batch_vals, next_values, GAMMA, LAMBDA)[0]
        # GAE=torch.tensor(GAE,dtype=torch.float).to(device).reshape(-1)
        #     GAE.append(gae)
        # GAE = GAE[::-1]
        # GAE = torch.tensor(GAE, dtype=torch.float).to(device).reshape(-1)
  
        _, next_value, = model(batch_obs[-1,:,:,:].reshape(1,1,84,84))
        next_value = next_value.squeeze()
        gae = 0
        R = []
        for value, reward, done in list(zip(batch_vals, rewards, dones))[::-1]:
            gae = gae * GAMMA * LAMBDA *(1-done)
            # ic(value.device)
            gae = gae + reward + GAMMA * next_value.item() * (1 - done) - value.item()
            
            next_value = value
            R.append(gae + value.item())
        R = R[::-1]
        R = torch.tensor(R, dtype=torch.float).to(device).reshape(-1)
        advantages = R - batch_vals
  
        # ic(GAE.shape)
        # ic(batch_vals.shape)
        # ic(len(rewards))
        # ic(next_values.shape)
        # ic(GAE[:30])#很大
        # ic(batch_vals[:10])
        # ic(rewards[:10])
        # ic(next_values[:10])
        # ic(batch_vals[:10])
        for nepoch in range(N_EPOCHS):
            # for batch_obs, act, old_log_probs, rtg, lens in zip(batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens):
            # ic(batch_obs.shape,batch_acts.shape,batch_log_probs.shape,batch_rtgs.shape)
            # ic(batch_rtgs[:20])
            # ic(batch_obs[:20])
            logits, value = model(batch_obs)
            # _,value=critic(batch_obs)
            policy = F.softmax(logits, dim=1)
            m = Categorical(policy)
            # ic(batch_acts.device)
            new_log_probs = m.log_prob(batch_acts)
            ratio = torch.exp(new_log_probs - batch_log_probs)
            # ic(policy[:20])
            # ic(new_log_probs[:20])
            # ic(logits.shape,policy.shape,new_log_probs.shape,ratio.shape)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * advantages
            entropy_loss = (m.entropy()).mean()
            
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            # critic_loss = F.smooth_l1_loss(reward+GAMMA*next_values, value.squeeze())
            critic_loss = F.smooth_l1_loss(R, value.squeeze())
            
            loss = critic_loss + actor_loss - ENT_COEF * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'actor_loss:{actor_loss.item()},critic_loss:{critic_loss.item()},entropy_loss:{entropy_loss.item()}')
        # if epoch % 10 == 0:
        torch.save(model,model_path)
        
def test():
    state = env.custom_reset()
    env.render=True
    '''
    一次从头到尾跑完一趟游戏, 多趟组成一个batchs
    '''
    batch_obs, batch_acts, batch_log_probs,batch_rews, batch_rtgs, batch_lens,batch_vals=[],[],[],[],[],[],[]
    
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
        batch_obs.append(state)
        logits,_=actor(torch.FloatTensor(state).to(device))
        # print(logits)
        batch_vals.append(value.detach())
        policy=F.softmax(logits.detach(),dim=1)
        #得到policy中,值最大的下标
        action=torch.argmax(policy)
        state,reward,done,info=env.custom_step(action.item())
        time.sleep(0.5)

# if __name__ == '__main__':
#     fire.Fire()
train()
test()
