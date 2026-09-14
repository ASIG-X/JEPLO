import gymnasium as gym

# flake8: noqa F401
from . import go2_loco_env, go2_loco_ppo_cfg

gym.register(
    id="Unitree-Go2-Locomotion",
    entry_point=f"{__name__}.go2_loco_env:Go2LocoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_loco_env:Go2LocoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.go2_loco_ppo_cfg:Go2LocoPPOCfg",
    },
)
