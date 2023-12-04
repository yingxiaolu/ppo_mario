#The 1-1 specifies the map to be loaded
 # Standar versión
#STAGE_NAME = 'SuperMarioBros-1-1-v3' # Rectangle versión
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,RIGHT_ONLY,COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from icecream import ic
import cv2
import gym
import numpy as np
import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import gym_super_mario_bros
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from icecream import ic
import pdb

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        gym.ObservationWrapper.__init__(self, env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (size, size, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:,:,None]
        return frame

class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        return self.env.reset(**kwargs)
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += max(0, info['x_pos'] - self.max_x)
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info["flag_get"]:
            reward += 500
            done = True
            print("GOAL")
        if info["life"] < 2:
            reward -= 500
            done = True
        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

class Customenv():
    def __init__(self,render=False):
        STAGE_NAME = 'SuperMarioBros-1-3-v0'
        MOVEMENT = [["A"],["right","A","B"],["right","A"],["left","A"],["NOOP"]]
        env = gym_super_mario_bros.make(STAGE_NAME)
        env = JoypadSpace(env, MOVEMENT)
        self.info=env.unwrapped._get_info()
        self.origin_info=self.info
        # env = CustomRewardAndDoneEnv(env)
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeEnv(env, size=84)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')
        self.render=render
        self.env=env
        self.state=self.env.reset()

        self._reinit__()
        
    def custom_reset(self):
        self.state=self.env.reset()
        # state=state[:,:,0]*0.299+state[:,:,1]*0.587+state[:,:,2]*0.114
        # self.state=state
        self.info=self.origin_info
        self._reinit__()
        return self.state
        
    def _reinit__(self):
        # self.reward=0
        self.done=False
        self.score=0
        self.coins=0
        self.time=self.info['time']
        # self.deltatime=0
        # self.coins=self.info['coins']
        # self.deltacoins=0
        self.x_pos=self.info['x_pos']
        self.max_x_pos=self.x_pos
        self.y_pos=self.info['y_pos']
        # self.max_rew=5
        # self.min_rew=-5
        # self.delta_x=0
    
    def custom_step(self,action):
        self.state,reward,self.done,info=self.env.step([action])
        info=info[0]
        rew=reward.item()
        reward=0
        # state=state[:,:,0]*0.299+state[:,:,1]*0.587+state[:,:,2]*0.114
        # self.state=state
        if self.render:
            self.env.render()
        deltatime=info['time']-self.time
        self.time=info['time']
        delta_coins=info['coins']-self.coins
        self.coins=info['coins']
        delta_x=info['x_pos']-self.x_pos
        self.x_pos=info['x_pos']
        delta_score=info['score']-self.score
        self.score=info['score']
        # reward=deltacoins*10+deltatime+delta_x
        exceed_xpos=max(0,info['x_pos']-self.max_x_pos)
        delta_y=info['y_pos']-self.y_pos
        self.y_pos=info['y_pos']
        self.max_x_pos=max(self.max_x_pos,info['x_pos'])
        reward=-0.3
        reward+=exceed_xpos
        # reward+=delta_y*4 if 0<delta_y<20 else 0#鼓励高跳
        reward+=delta_coins if delta_coins>0 else 0
        reward+=delta_score if delta_score>0 else 0
        # ic(reward,exceed_xpos,delta_y,delta_coins,delta_score)
        if info['flag_get']:
            reward+=100
        # if info['x_pos']>self.max_x_pos:
        #     self.max_x_pos=info['x_pos']
        # reward+=np.log(info['x_pos'])/40 if info['x_pos']!=0 else 0 #相同图像应该采取相同措施, 不应该加这个, 和动作不相关
        if self.done and info['time']>0 and info['flag_get']==False: #摔死的,碰死的
            reward-=100
        # if self.done and info['time']==0: #如果时间到了但是没到终点
        #     reward-=10
        # if delta_x<=0.0:#如果没动
        #     reward-=10
        
        # if reward>1:
        #     reward=np.log(reward)
        # elif reward<-1:
        #     reward=-np.log(-reward)*10#往前波及几帧
            
        reward=reward*0.1
        # reward=np.log(reward)/10 if reward>0.001 else 0
        # ic(reward)
        self.info=info
        return self.state,reward,self.done,info
    
    def custom_render(self): 
        self.env.render()
    
    def custom_close(self):
        self.env.close()