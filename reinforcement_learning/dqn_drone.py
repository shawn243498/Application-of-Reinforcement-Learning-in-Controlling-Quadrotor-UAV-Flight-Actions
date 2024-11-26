import gym
import time
import yaml

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback

# Get train environment configs
with open('scripts/config_train.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# Create a DummyVecEnv
env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:airsim-env-v0",   # v0 is for TRT, v1 is normal
        ip_address="127.0.0.1",
        image_shape=(240,360,3),
        env_config=env_config["TrainEnv"],
    )
)])

# Wrap env as VecTransposeImage (Channel last to channel first)
env = VecTransposeImage(env)

# Initialize DQN
model = DQN(
    'CnnPolicy',
    env,
    verbose=1,
    learning_rate=0.00025,
    buffer_size=10000,
    device="cuda",
    seed=1,
    tensorboard_log="./tb_logs/",
)

# Evaluation callback
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=10,
    best_model_save_path=".",
    log_path=".",
    eval_freq=1000,
)

callbacks.append(eval_callback)
kwargs = {}
kwargs["callback"] = callbacks

log_name = "dqn_run_" + str(time.time())

model.learn(
    total_timesteps=150000,
    tb_log_name=log_name,
    **kwargs
)

# Save policy weights
model.save("dqn_navigation_policy")
