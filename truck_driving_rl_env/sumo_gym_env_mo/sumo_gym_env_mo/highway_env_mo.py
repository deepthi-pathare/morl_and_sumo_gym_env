import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sumo_gym_env.envs.highway_env import Highway, CHANGE_LANE_LEFT, CHANGE_LANE_RIGHT
import math
import traci
import wandb

class Highway_MO(Highway):
    """
    This class creates a gym-like highway driving environment for multi-objective RL.

    Args:
        sim_params: Parameters that describe the simulation setup, the action space, and the reward function
        road_params: Parameters that describe the road geometry, rules, and properties of different vehicles
        use_gui (bool): Run simulation w/wo GUI
        start_time (str): Optional label
    """
    def __init__(self, sim_params, road_params, use_gui=True, env_label=None):
        super(Highway_MO, self).__init__(sim_params, road_params, use_gui, env_label)
        
        self.reward_dim = 3 # Outside road/collision penalty, driver cost
        self.reward_space = spaces.Box(
            low=np.array([-self.collision_penalty, -math.inf, -math.inf]),
            high=np.array([self.completion_reward, 0, 0]),
            shape=(self.reward_dim,),
            dtype=np.float64,
        )
        
    def reward_model(self, new_state, action, ego_collision=False, ego_near_collision=False, outside_road=False):
        """
        Reward model of the highway environment.

        Args:
            new_state (list) : New state of the vehicles
            action (int): Action by the agent
            ego_collision (bool): True if ego vehicle collides.
            ego_near_collision (bool): True if ego vehicle is close to a collision.
            outside_road (bool): True if ego vehicle drives off the road.

        Returns:
            reward (float): Reward for the current environment step.
        """
        reward = np.zeros(self.reward_dim)

        # Collision/Near collision/Outside road penalty
        if outside_road or ego_collision:
            reward[0] -= self.collision_penalty
        elif new_state[4][self.ego_index] == "exit":
            reward[0] += self.completion_reward
        
        if action == CHANGE_LANE_LEFT or action == CHANGE_LANE_RIGHT:
            electricity_consumed = self.lat_controller.energy_consumed # kwh
            time_consumed = self.lane_change_duration / 3600 # hours
        else:
            electricity_consumed = self.long_controller.energy_consumed # kwh
            time_consumed = self.long_control_duration / 3600 # hours

        driver_cost = (time_consumed * self.driver_cost)
        energy_cost = (electricity_consumed * self.electricity_cost)

        reward[1] -= (driver_cost) 
        reward[2] -= (energy_cost)

        self.ego_speed += new_state[1][self.ego_index, 0] # This is to compute the average speed.
        self.ego_energy_cost += (electricity_consumed * self.electricity_cost)
        self.ego_driver_cost += (time_consumed * self.driver_cost)

        return reward