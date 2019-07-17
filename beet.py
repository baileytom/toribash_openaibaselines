import gym
import torille.envs

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2


# Enjoy trained agent
if __name__ == '__main__':

	# multiprocess environment
	n_cpu = 4
	env = SubprocVecEnv([lambda: gym.make('Toribash-Custom-v0') for i in range(n_cpu)])

	model = PPO2(MlpPolicy, env, verbose=1)
	model = PPO2.load("ppo2_cartpole")

	obs = env.reset()
	while True:
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		#env.render()