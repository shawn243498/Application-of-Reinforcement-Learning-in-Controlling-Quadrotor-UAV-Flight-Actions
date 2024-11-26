import gym
import time
import yaml

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

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

# Initialize A2C
model = A2C(
    'CnnPolicy',
    env,
    verbose=1,
    device="cuda",
    seed=1,
    policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)),
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

log_name = "a2c_run_" + str(time.time())

model.learn(
    total_timesteps=150000,
    tb_log_name=log_name,
    **kwargs
)

# Save policy weights
model.save("a2c_navigation_policy")
