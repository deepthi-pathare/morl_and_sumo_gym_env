import gymnasium as gym
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
import env_parameters as ps
import sumo_gym_env_mo

def make_env(env_id, seed, idx, gamma):
    """Function to create vectorized environments. 

    Args:
        env_id: Environment ID (for MO-Gymnasium)
        seed: Seed
        idx: Index of the environment
        gamma: Discount factor
    """
    def thunk():
        render_mode = None
        if idx == 0:
            render_mode = "rgb_array"
        if "sumo_highway_env" in env_id:
            env = mo_gym.make(env_id, sim_params=ps.sim_params, road_params=ps.road_params, use_gui=False, env_label = "training")
        else:
            env = mo_gym.make(env_id, render_mode=render_mode)
        reward_dim = env.unwrapped.reward_space.shape[0]
     
        # Check if action space is discrete
        if hasattr(env.action_space, 'n'):  # Discrete action space
            # Don't apply ClipAction wrapper for discrete action spaces
            pass
        else:  # Continuous action space
            env = gym.wrappers.ClipAction(env)
            
        env = MORecordEpisodeStatistics(env, gamma=gamma)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk