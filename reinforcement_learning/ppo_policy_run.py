import gym
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


# Get train environment configs
with open('scripts/config_test.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# print(env_config)
# Create a DummyVecEnv
env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:test-env-v1",    # v0 is for TRT, v1 is normal
        ip_address="127.0.0.1", 
        image_shape=(240,360,3),
        env_config=env_config["TrainEnv"]
    )
)])

# Wrap env as VecTransposeImage (Channel last to channel first)
env = VecTransposeImage(env)

# Load an existing model
model = PPO.load(env=env, path="ppo_navigation_policy")

stop = False

# Run the trained policy
obs = env.reset()
while True:
    # print(obs)
    action, _ = model.predict(obs, deterministic=True)
    obs, stop, dones, info = env.step(action)

    if stop[0] == 1.0:
        break