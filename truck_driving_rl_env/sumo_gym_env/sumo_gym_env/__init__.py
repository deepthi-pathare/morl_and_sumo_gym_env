from gymnasium.envs.registration import register

register(
    id='sumo_highway_env-v1',
    entry_point='sumo_gym_env.envs:Highway',
)