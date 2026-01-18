from gymnasium.envs.registration import register

register(
    id='sumo_highway_env_mo-v0',
    entry_point='sumo_gym_env_mo.highway_env_mo:Highway_MO',
)