from .airsim_env_TRT import AirSimDroneEnv_TRT, TestEnv_TRT
from .airsim_env import AirSimDroneEnv, TestEnv
from gym.envs.registration import register

# Register AirSim environment as a gym environment
register(
    id="airsim-env-v1", entry_point="scripts:AirSimDroneEnv",
)

# Register AirSim environment as a gym environment
register(
    id="test-env-v1", entry_point="scripts:TestEnv",
)

# Register AirSim environment as a gym environment
register(
    id="airsim-env-v0", entry_point="scripts:AirSimDroneEnv_TRT",
)

# Register AirSim environment as a gym environment
register(
    id="test-env-v0", entry_point="scripts:TestEnv_TRT",
)
