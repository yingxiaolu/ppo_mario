from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from PIL import Image
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from icecream import ic
import cv2
import gym
import numpy as np
import time
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
    
STAGE_NAME = 'SuperMarioBros-1-1-v0'
env = gym_super_mario_bros.make(STAGE_NAME)
MOVEMENT = [["A"],["right","A","B"],["right","A"],["left","A"],["NOOP"]]
env = JoypadSpace(env,MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = ResizeEnv(env, size=84)
done = True
reward_list=[]
x_pos_list=[]
time_list=[]
for _ in range(100):
    for step in range(500000):
        if done: # Done will be true if Mario dies in the game
            state = env.reset()
        action=env.action_space.sample() # Take a random action
        # ic(action)
        state, reward, done, info = env.step(action)
        ic(action,state.shape, reward, done, info)
        reward_list.append(reward)
        x_pos_list.append(info['x_pos'])
        time_list.append(info['time'])
        # ic(state.shape, reward, done, info)
        # _=input()
        # time.sleep(0.1)
        env.render() # If we are running the program in Colab we will need to comment the rendering of the environment. 
        if done:
            break
env.close()
print(reward_list[-100:])
print(x_pos_list[-100:])
print(time_list[-100:])