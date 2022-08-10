import gym
import Heat

env = gym.make('Heat_d-v0')
for _ in range(2):
    action = env.action_space.sample()   # 从动作空间中随机选取一个动作
    observation, reward, done, info = env.step(action)  # 用于提交动作，括号内是具体的动作
    print(reward)
env.close()