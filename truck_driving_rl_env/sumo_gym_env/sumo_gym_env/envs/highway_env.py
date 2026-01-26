import os
import sys
import numpy as np
import copy
import warnings
import gymnasium as gym
import math
from gymnasium import spaces

#warnings.simplefilter('always', UserWarning)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

from sumo_gym_env.common.road import Road
from sumo_gym_env.common.longitudinal_controller import Longitudinal_Controller
from sumo_gym_env.common.lateral_controller import Lateral_Controller
from sumo_gym_env.common.collision import check_for_collision

# Action values
ACTION_COUNT = 8
SET_TARGET_VEH_SHORT_GAP = 0
SET_TARGET_VEH_MEDIUM_GAP = 1
SET_TARGET_VEH_LONG_GAP = 2
INCREASE_DESIRED_SPEED = 3
DECREASE_DESIRED_SPEED = 4
CHANGE_LANE_LEFT = 5
CHANGE_LANE_RIGHT = 6
MAINTAIN_SPEED_AND_GAP = 7

# Sumo subscription constants
POSITION = 66
LONG_SPEED = 64
LAT_SPEED = 50
LONG_ACC = 114
LANE_INDEX = 82
INDICATOR = 91 # Named as signal states in sumo
ROAD_ID = 80 # Edge name

class Highway(gym.Env):
    """
    This class creates a gym-like highway driving environment.

    Args:
        sim_params: Parameters that describe the simulation setup, the action space, and the reward function
        road_params: Parameters that describe the road geometry, rules, and properties of different vehicles
        use_gui (bool): Run simulation w/wo GUI
        start_time (str): Optional label
    """
    def __init__(self, sim_params, road_params, use_gui=True, env_label=None):
        super(Highway, self).__init__()
        self.step_ = 0
        self.env_label = env_label
        self.sim_step_length = sim_params['sim_step_length']
        self.long_control_duration = sim_params['long_control_duration']
        self.max_steps = sim_params['max_steps']
        self.init_steps = sim_params['init_steps']
        #self.nb_vehicles = sim_params['nb_vehicles']
        self.vehicles = None
        self.safety_check = sim_params['safety_check']

        self.road = Road(road_params)
        self.road.create_road()

        # Traffic properties
        self.traffic_density = sim_params['traffic_density']
        self.road_length = sim_params['road_length']
        self.moving_window_size = sim_params['moving_window_size'] #moving window will be +/-(size/2) from ego position
        self.spawning_safe_dist = sim_params['spawning_safe_dist']
        self.hvr_cars = sim_params['hvr_cars']
        self.hvr_trucks = sim_params['hvr_trucks']

        # Vehicle count
        nb_surr_vehicles = round(sim_params['traffic_density'] * sim_params['moving_window_size']) # Number of inserted vehicles.
        self.nb_cars = round(nb_surr_vehicles * sim_params['hvr_cars'])
        self.nb_trucks = round(nb_surr_vehicles * sim_params['hvr_trucks']) + 1 # +1 for ego vehicle
        self.nb_vehicles = self.nb_cars + self.nb_trucks

        print("No of vehicles (total, cars, truck):", self.nb_vehicles, self.nb_cars, self.nb_trucks)

        # Desired speed distribution
        self.speed_mean_cars = sim_params['speed_mean_cars']
        self.speed_std_cars = sim_params['speed_std_cars']
        self.speed_mean_trucks = sim_params['speed_mean_trucks']
        self.speed_std_trucks = sim_params['speed_std_trucks']

        self.nb_lanes = self.road.road_params['nb_lanes']
        self.lane_width = self.road.road_params['lane_width']
        self.lane_change_duration = self.road.road_params['lane_change_duration']
        self.positions = np.zeros([self.nb_vehicles, 2])
        self.speeds = np.zeros([self.nb_vehicles, 2])
        self.accs = np.zeros([self.nb_vehicles, 1])
        self.lanes = np.zeros([self.nb_vehicles])
        self.indicators = np.zeros([self.nb_vehicles])
        self.edge_name = np.empty([self.nb_vehicles], dtype='object')
        self.veh_lengths = np.zeros([self.nb_vehicles])
        self.veh_widths = np.zeros([self.nb_vehicles])
        self.init_ego_position = 0.
        self.ego_index = self.nb_vehicles // 2
        self.ego_id = 'veh' + str(self.ego_index).zfill(int(np.ceil(np.log10(self.nb_vehicles)))) 

        self.max_speed = self.road.road_params['speed_range'][1]
        self.min_speed = self.road.road_params['speed_range'][0]
        self.max_allowed_ego_speed = min(self.road.road_params['vehicles'][0]['maxSpeed'], road_params['max_road_speed'])
        self.min_allowed_ego_speed = road_params['min_road_speed']

        self.sensor_range = sim_params['sensor_range']
        self.sensor_nb_vehicles = min(sim_params['sensor_nb_vehicles'], self.nb_vehicles)
        self.target_veh_short_gap = sim_params['target_veh_short_gap']
        self.target_veh_medium_gap = sim_params['target_veh_medium_gap']
        self.target_veh_long_gap = sim_params['target_veh_long_gap']
        self.cruise_acceleration = sim_params['cruise_acceleration']
        self.cruise_deceleration = sim_params['cruise_deceleration']

        # Rewards/Penalties
        self.collision_penalty = sim_params['collision_penalty']
        self.near_collision_penalty = sim_params['near_collision_penalty']
        self.outside_road_penalty = sim_params['outside_road_penalty']
        self.lane_change_penalty = sim_params['lane_change_penalty']
        self.completion_reward = sim_params['completion_reward']
        self.electricity_cost = sim_params['electricity_cost']
        self.driver_cost = sim_params['driver_cost']
        self.nb_ego_states = 9
        self.nb_states_per_vehicle = 9

        # Actionspace and observation space for gym
        state_size = self.nb_ego_states + (self.sensor_nb_vehicles * self.nb_states_per_vehicle)
        l = np.full(shape=(state_size,), fill_value=(-math.inf))
        h = np.full(shape=(state_size,), fill_value=(math.inf))
        self.observation_space = spaces.Box(low = l, 
                                            high = h,
                                            dtype = np.float64)
    
        # Define an action space ranging from 0 to ACTION_COUNT-1
        self.action_space = spaces.Discrete(ACTION_COUNT)

        # Evaluation metrices
        self.nb_collisions = 0
        self.nb_near_collisions = 0
        self.nb_outside_road = 0
        self.nb_max_step = 0
        self.nb_max_distance = 0

        self.momentum = []
        self.kinetic_energy = []
        self.episode_no = 0
        self.total_timestep = 0
        
        # Low level controllers
        self.long_controller = Longitudinal_Controller(self.ego_id, self.max_allowed_ego_speed, self.min_allowed_ego_speed,
                                                       self.target_veh_long_gap, self.sim_step_length, self.long_control_duration)
        self.lat_controller = Lateral_Controller(self.ego_id, self.nb_lanes, self.lane_width, self.sim_step_length)

        self.ego_speed = 0
        self.ego_energy_cost = 0
        self.ego_driver_cost = 0

        max_traffic_density = self.nb_lanes / (road_params['vehicles'][0]['length'] + self.spawning_safe_dist)
        if self.traffic_density > max_traffic_density:
            print("Traffic density cannot exceed max_traffic_density vehicles per m. This restriction concerns with the maximum vehicles that can be accomodated in the moving window with a safe distance.")
            return
        
        # launch sumo
        self.use_gui = use_gui
        if self.use_gui:
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')
        
        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        if sim_params['remove_sumo_warnings']:
            cmd = [sumo_binary, "-c", self.road.road_path + self.road.name + ".sumocfg", "--start", "--step-length", str(self.sim_step_length), "--no-warnings"]
        else:
            cmd = [sumo_binary, "-c", self.road.road_path + self.road.name + ".sumocfg", "--start", "--step-length", str(self.sim_step_length)]

        traci.start(cmd, label=env_label)

    def reset(self, seed = None, options=None):
        """
        Resets the highway driving environment to a new random initial state.

        The ego vehicle starts in a random lane. A number of surrounding vehicles are added to random positions.
        Vehicles in front of the ego vehicle are initalized with a lower speed than the ego vehicle, and vehicles behind
         the ego vehicle are initalized with a faster speed. If two vehicles vehicles are initalized too close
         to each other, one of them is moved.

        Args:
            seed (None): The seed that is used to initialize the environment.

        Returns:
            observation (ndarray): The observation of the traffic situation, according to the sensor model.
        """
        self.episode_no += 1
        self.total_timestep += self.step_
        np.random.seed(seed)

        traci.switch(self.env_label)

        # Remove all vehicles
        for veh in traci.vehicle.getIDList():
            traci.vehicle.unsubscribe(veh)
            traci.vehicle.remove(veh)
        traci.simulationStep()

        # Randomly distribute the order of adding vehicle types
        veh_type = self.nb_trucks * ["truck"] + self.nb_cars * ["car"]
        np.random.shuffle(veh_type)

        # Identify the first truck and swap it to the middle position
        truck_indices = [i for i, v in enumerate(veh_type) if v == "truck"]
        middle_index = self.ego_index
        if truck_indices:
            first_truck_idx = truck_indices[0]
            veh_type[first_truck_idx], veh_type[middle_index] = veh_type[middle_index], veh_type[first_truck_idx]

        #print("Vehicles inserted in the order: ", veh_type)

        init_speed = np.zeros(self.nb_vehicles)

        # Insert vehicles
        for i, type in enumerate(veh_type):
            # Assign ID
            veh_id = 'veh' + str(i).zfill(int(np.ceil(np.log10(self.nb_vehicles))))   # Add leading zeros to number

            # Set initial speed of vehicles
            if type == "truck":
                init_speed[i] = round(np.random.normal(self.speed_mean_trucks, self.speed_std_trucks), 2)
            else:
                init_speed[i] = round(np.random.normal(self.speed_mean_cars, self.speed_std_cars), 2)

            lane = i % self.nb_lanes
            traci.vehicle.add(veh_id, 'route0', typeID=type, depart=None, departLane=lane,
                              departPos='base', departSpeed=self.road.road_params['vehicles'][0]['maxSpeed'],
                              arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='',
                              line='', personCapacity=0, personNumber=0)
            if (i + 1) % self.nb_lanes == 0:  # When all lanes are filled
                traci.simulationStep()  # Deploy vehicles
                # Move all vehicles some meters forward so that next vehicles to be added do not get intersected with them
                # (Refer constraints for adding vehicles: https://sumo.dlr.de/docs/Simulation/VehicleInsertion.html )
                for veh in traci.vehicle.getIDList():
                    traci.vehicle.moveTo(veh, traci.vehicle.getLaneID(veh), traci.vehicle.getLanePosition(veh) + self.spawning_safe_dist)

        traci.simulationStep()
        assert (len(traci.vehicle.getIDList()) == self.nb_vehicles)
        self.vehicles = traci.vehicle.getIDList()

        # Position the vehicles and set their speeds as above using traci
        for i, veh in enumerate(self.vehicles):
            traci.vehicle.setPreviousSpeed(veh, init_speed[i])   # Set current speed
            traci.vehicle.setMaxSpeed(veh, init_speed[i])   # Set speed of "cruise controller"

        # Variable subscriptions from sumo
        for veh in self.vehicles:
            traci.vehicle.subscribe(veh, [POSITION, LONG_SPEED, LAT_SPEED, LONG_ACC, LANE_INDEX, INDICATOR, ROAD_ID])

        # Initial simulation steps
        for i in range(self.init_steps):
            traci.simulationStep()

        # Turn off all internal lane changes and all safety checks for ego vehicle
        if not self.safety_check:
            traci.vehicle.setSpeedMode(self.ego_id, 0)
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)

        # Return speed control of surrounding vehicles to sumo, starting from initial random speed
        for veh in self.vehicles:
            if veh != self.ego_id:
                traci.vehicle.setSpeed(veh, -1)
                traci.vehicle.setLaneChangeMode(veh, 512)

        if self.use_gui:
            traci.gui.trackVehicle('View #0', self.ego_id)

        self.step_ = 0
        self.ego_speed = 0
        self.ego_energy_cost = 0
        self.ego_driver_cost = 0
        self.momentum = []
        self.kinetic_energy = []

        self.init_ego_position = traci.vehicle.getPosition(self.ego_id)[0]

        # Get position and speed of vehicles after initial simulation steps
        for i, veh in enumerate(self.vehicles):
            out = traci.vehicle.getSubscriptionResults(veh)
            # Lateral positions from sumo (-8, -4.8, -1.6) are converted to (0, 3.2, 6.4) where 0 is the middle of rightmost lane.
            self.positions[i, :] = np.array(out[POSITION]) + np.array([0, self.lane_width * self.nb_lanes -
                                                                       self.lane_width/2])
            self.positions[i, 1] = round(self.positions[i, 1], 10) #in floating-point arithmetic, above operation gives 1.77635684e-15 instead of 0. so rounded.

            self.speeds[i, 0] = out[LONG_SPEED]
            self.speeds[i, 1] = out[LAT_SPEED]
            self.accs[i] = out[LONG_ACC]
            self.lanes[i] = out[LANE_INDEX]
            self.indicators[i] = out[INDICATOR]
            self.edge_name[i] = out[ROAD_ID]
            self.veh_lengths[i] = traci.vehicle.getLength(veh)
            self.veh_widths[i] = traci.vehicle.getWidth(veh)

            # Set color to vehicles based on speed
            if self.use_gui:
                if i == self.ego_index:
                    traci.vehicle.setColor(veh, (0, 200, 0))
                else:
                    speed_factor = (self.speeds[i, 0] - self.min_speed)/(self.max_speed - self.min_speed)
                    speed_factor = np.max([speed_factor, 0])
                    speed_factor = np.min([speed_factor, 1])
                    traci.vehicle.setColor(veh, (255, int(255*(1-speed_factor)), 0))

        # Create observation space
        state = [self.positions, self.speeds, self.lanes, self.indicators, self.edge_name, self.veh_lengths, self.veh_widths, False]
        observation = self.sensor_model(state)

        #self.ego_energy_cost += (self.long_controller.get_initial_kinetic_energy(self.speeds[self.ego_index,0]) * self.electricity_cost)

        self.compute_kinematics()

        if self.use_gui:
            self.print_info_in_gui()

        info = {}
        return observation, info
    
    def step(self, action, action_info=None):
        """
        Transition the environment to the next state with the specified action.

        Args:
            action (int): Specified action, which is then translated to a longitudinal and lateral action.
            action_info (dict): Only used to display information in the GUI.

        Returns:
            tuple, containing:
                observation (ndarray): Observation of the environment, given by the sensor model.
                reward (float): Reward of the current time step.
                done (bool): True if terminal state is reached, otherwise False
                info (list): List of information on what caused the terminal condition.

        """
        traci.switch(self.env_label)

        self.step_ += 1
        outside_road = False 
        done = False
        truncated = False       
        info = []
        ego_collision, ego_near_collision, other_veh_collision = False, False, False
        prev_ego_pos = self.positions[0, 0]

        if action == SET_TARGET_VEH_SHORT_GAP:
                ego_collision, ego_near_collision, other_veh_collision = self.long_controller.change_desired_timegap(self.target_veh_short_gap)
        elif action == SET_TARGET_VEH_MEDIUM_GAP:
                ego_collision, ego_near_collision, other_veh_collision = self.long_controller.change_desired_timegap(self.target_veh_medium_gap)
        elif action == SET_TARGET_VEH_LONG_GAP:
                ego_collision, ego_near_collision, other_veh_collision = self.long_controller.change_desired_timegap(self.target_veh_long_gap)
        elif action == INCREASE_DESIRED_SPEED:
                ego_collision, ego_near_collision, other_veh_collision = self.long_controller.change_desired_speed(self.cruise_acceleration)
        elif action == DECREASE_DESIRED_SPEED:
                ego_collision, ego_near_collision, other_veh_collision = self.long_controller.change_desired_speed(self.cruise_deceleration)
        elif action == CHANGE_LANE_LEFT:
                if(self.lanes[self.ego_index] == (self.nb_lanes - 1)):
                    outside_road = True
                else:
                    ego_collision, ego_near_collision, other_veh_collision = self.lat_controller.change_to_left_lane()
        elif action == CHANGE_LANE_RIGHT:
                if(self.lanes[self.ego_index] == 0):
                    outside_road = True
                else:
                    ego_collision, ego_near_collision, other_veh_collision = self.lat_controller.change_to_right_lane()
        elif action == MAINTAIN_SPEED_AND_GAP:
                ego_collision, ego_near_collision, other_veh_collision = self.long_controller.maintain_speed_and_gap()                
        else:
                print('Undefined action, this should never happen')

        self.vehicles = traci.vehicle.getIDList()
        
        # If any vehicles have reached exit edge, reroute them to a moving window around the ego truck to maintain a fixed traffic density
        self.reroute_vehicles()

        self.vehicles = traci.vehicle.getIDList()

        # Number of digits in vehicle name. Can't just enumerate index because vehicles can be removed in the event of
        # simultaneous change to center lane.
        nb_digits = int(np.floor(np.log10(self.nb_vehicles))) + 1
        for veh in self.vehicles:
            i = int(veh[-nb_digits:])   # See comment above
            out = traci.vehicle.getSubscriptionResults(veh)
            # Skip if the vehicle has left from simulation
            if out == {}:
                continue
            # Lateral positions from sumo (-8, -4.8, -1.6) are converted to (0, 3.2, 6.4) where 0 is the middle of rightmost lane.
            self.positions[i, :] = np.array(out[POSITION]) + np.array([0, self.lane_width * self.nb_lanes -
                                                                       self.lane_width/2])
            self.positions[i, 1] = round(self.positions[i, 1], 10) #in floating-point arithmetic, above operation gives 1.77635684e-15 instead of 0. so rounded.
            self.speeds[i, 0] = out[LONG_SPEED]
            self.speeds[i, 1] = out[LAT_SPEED]
            self.accs[i] = out[LONG_ACC]
            self.lanes[i] = out[LANE_INDEX]
            self.indicators[i] = out[INDICATOR]
            self.edge_name[i] = out[ROAD_ID]
            self.veh_lengths[i] = traci.vehicle.getLength(veh)
            self.veh_widths[i] = traci.vehicle.getWidth(veh)

            # Set color to vehicles based on speed
            if self.use_gui and not i == self.ego_index:
                speed_factor = (self.speeds[i, 0] - self.min_speed)/(self.max_speed - self.min_speed)
                speed_factor = np.max([speed_factor, 0])
                speed_factor = np.min([speed_factor, 1])
                traci.vehicle.setColor(veh, (255, int(255*(1-speed_factor)), 0))

        new_ego_pos = self.positions[0, 0]
        dist_per_step = new_ego_pos - prev_ego_pos

        if other_veh_collision:            
            warnings.warn('Collision not involving ego vehicle. This should normally not happen.')

        if outside_road:
            self.nb_outside_road += 1
            done = True
            info.append('Terminated due to driving outside road')
            #print(info, action)
        elif ego_collision:
            self.nb_collisions += 1
            done = True
            info.append('Terminated due to collision')
            #print(info, action)
        elif ego_near_collision:
            self.nb_near_collisions += 1
            info.append('Near collision occured')
            #print(info)
        elif self.edge_name[self.ego_index] == "exit":
            self.nb_max_distance += 1
            done = True
            info.append('Reached exit edge')
        elif self.step_ == self.max_steps:
            self.nb_max_step += 1
            truncated = True
            info.append('Truncated as maximum steps reached')
        else:
            pass

        state = copy.deepcopy([self.positions, self.speeds, self.lanes, self.indicators, self.edge_name, self.veh_lengths, self.veh_widths, done])
        observation = self.sensor_model(state)
        reward = self.reward_model(state, action, ego_collision, ego_near_collision, outside_road)

        self.compute_kinematics()

        if self.use_gui:
            self.print_info_in_gui()

        info_dict = {'info':info}
        return observation, reward, done, truncated, info_dict
    
    def compute_kinematics(self):
        kinetic_energy = 0
        momentum = 0

        mov_window_start = max(0, self.positions[self.ego_index][0] - (self.moving_window_size / 2))
        mov_window_end = min(self.positions[self.ego_index][0] + (self.moving_window_size / 2), self.road_length)

        nb_digits = int(np.floor(np.log10(self.nb_vehicles))) + 1
        for veh in self.vehicles:
            i = int(veh[-nb_digits:])
            if self.positions[i][0] >= mov_window_start and self.positions[i][0] <= mov_window_end:
                mass = 4000 if i < self.nb_trucks else 1500
                kinetic_energy += (0.5 * mass * self.speeds[i, 0] * self.speeds[i, 0])
                momentum += (mass * self.speeds[i, 0])

        self.kinetic_energy.append(kinetic_energy/ (1000 * 3600))
        self.momentum.append(momentum)

    def get_edge_pos(self, pos):
        """
        Given a longitudinal position, get the corresponding edge name.

        Args:

        Returns:
        """
        edge_name = ""
        edge_rel_pos = 0     
        for i in range(0, len(self.road.road_params['edges'])):
            edge_start = self.road.road_params['nodes'][i][0] 
            edge_end = self.road.road_params['nodes'][i+1][0] 
            if pos >= edge_start and pos < edge_end:
                edge_name = self.road.road_params['edges'][i]
                edge_rel_pos = pos - edge_start
                return edge_name, edge_rel_pos
        
        print("Could not retrieve edge name for position ", pos)
        return None, None
    
    def reroute_vehicles(self):
        """
        Reroute veicles that reach the exit to maintain a fixed traffic density around the ego truck

        Args:

        Returns:
        """      
        mov_window_start = max(0, self.positions[self.ego_index][0] - (self.moving_window_size / 2))
        mov_window_end = min(self.positions[self.ego_index][0] + (self.moving_window_size / 2), self.road_length)

        #print('Mov window: {},{}'.format(mov_window_start,mov_window_end))

        rear_vehicles = []
        front_vehicles = []
        
        nb_digits = int(np.floor(np.log10(self.nb_vehicles))) + 1
        for veh in self.vehicles:
            i = int(veh[-nb_digits:])
            if i != self.ego_index:
                if self.positions[i][0] < mov_window_start:
                    rear_vehicles.append(veh)
                elif self.positions[i][0] > mov_window_end:
                    front_vehicles.append(veh)
                else:
                    continue
        
        # Add vehicles ahead of moving window to the beginning of window
        x_pos_rear = []
        for i, veh in enumerate(front_vehicles): 
            x_pos_rear.append(mov_window_start)
            if (i + 1) % self.nb_lanes == 0:  # When all lanes are filled
                # Move all vehicles a few meters backward. This is to keep a safe distance with existing vehicles in the window and new vehicles to be added.
                for i in range(0, len(x_pos_rear)):
                    x_pos_rear[i] = x_pos_rear[i] - self.spawning_safe_dist

        # Add vehicles behind the moving window to the end of window
        x_pos_front = []
        for i, veh in enumerate(rear_vehicles):
            x_pos_front.append(mov_window_end)
            if (i + 1) % self.nb_lanes == 0:  # When all lanes are filled
                # Move all vehicles a few meters forward. This is to keep a safe distance with existing vehicles in the window and new vehicles to be added.
                for i in range(0, len(x_pos_front)):
                    x_pos_front[i] = x_pos_front[i] + self.spawning_safe_dist

        for i, veh in enumerate(front_vehicles):  
            self.replace_veh(veh, x_pos_rear[i])
        
        for i, veh in enumerate(rear_vehicles):  
            self.replace_veh(veh, x_pos_front[i])

    def replace_veh(self, veh, x_pos):
        if x_pos < 0 or x_pos > self.road_length:
            # If new position for vehicle is outside given road segment, do not move them at this step.
            return
        
        lane = np.random.randint(0, self.nb_lanes)
        type = traci.vehicle.getTypeID(veh).split("@")[0]
        speed = traci.vehicle.getSpeed(veh)
        maxSpeed = traci.vehicle.getMaxSpeed(veh)
        traci.vehicle.remove(veh)
        
        edge_name, edge_rel_pos = self.get_edge_pos(x_pos)

        if edge_name is not None:
            traci.vehicle.add(veh, 'route0', typeID=type, depart=None, departLane=lane,
                              departPos='free', departSpeed='max',
                              arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='',
                              line='', personCapacity=0, personNumber=0)
            traci.vehicle.moveTo(veh, edge_name + '_' + str(lane), edge_rel_pos)

            traci.vehicle.setPreviousSpeed(veh, speed)   # Set current speed
            traci.vehicle.setMaxSpeed(veh, maxSpeed)   # Set speed of "cruise controller"

            #traci.vehicle.setSpeed(veh, -1)
        else:
            warnings.warn('Invalid edge name for the new position. Should not happen.')

    def reward_model(self, new_state, action, ego_collision=False, ego_near_collision=False, outside_road=False):
        """
        Reward model of the highway environment.

        Args:
            new_state (list) : New state of the vehicles
            action (int): Action by the agent
            ego_collision (bool): True if ego vehicle collides.
            ego_near_collision (bool): True if ego vehicle is close to a collision.
            outside_road (bool): True if ego vehicle drives off the road.
            outside_road (float): Distance travelled by ego vehicle in this RL step.
        Returns:
            reward (float): Reward for the current environment step.
        """    
        reward = 0

        if action == CHANGE_LANE_LEFT or action == CHANGE_LANE_RIGHT:
            electricity_consumed = self.lat_controller.energy_consumed # kwh
            time_consumed = self.lane_change_duration / 3600 
        else:
            electricity_consumed = self.long_controller.energy_consumed # kwh
            time_consumed = self.long_control_duration / 3600 

        reward -= (electricity_consumed * self.electricity_cost) 
        reward -= (time_consumed * self.driver_cost)
        
        # Collision/Near collision/Outside road penalty
        if outside_road:
            reward -= self.outside_road_penalty
        elif ego_near_collision:
            reward -= self.near_collision_penalty
        elif ego_collision:
            reward -= self.collision_penalty 
        # Reward when episode is completed successfully (ego vehicle reached exit edge)
        elif new_state[4][self.ego_index] == "exit":   
            reward += (self.completion_reward)

        # compute evaluation metrices
        self.ego_speed += new_state[1][self.ego_index, 0] # This is to compute the average speed.
        self.ego_energy_cost += (electricity_consumed * self.electricity_cost)
        self.ego_driver_cost += (time_consumed * self.driver_cost)
        return reward

    def sensor_model(self, state):
        """
        Sensor model of the ego vehicle.

        Creates an observation vector from the current state of the environment. All observations are normalized.
        Only surrounding vehicles within the sensor range are included.

        Args:
            state (list): Current state of the environment.

        Returns:
            observation( (ndarray): Current observation of the highway environment.
        """
        vehicles_in_range = np.abs(state[0][:, 0] - state[0][self.ego_index, 0]) <= self.sensor_range
        vehicles_in_range[self.ego_index] = False
        if np.sum(vehicles_in_range) > self.sensor_nb_vehicles:
            warnings.warn('More vehicles within range than sensor can represent')

        # Create a vector with state values(positionx,positiony lat speed, long speed) of each vehicle
        observation = np.zeros(self.nb_ego_states + self.nb_states_per_vehicle * self.sensor_nb_vehicles)
        observation[0] = state[0][self.ego_index, 0] # Longitudinal position
        observation[1] = state[1][self.ego_index, 0]   # Longitudinal speed
        observation[2] = np.sign(state[1][self.ego_index, 1])   # Lane change state
        observation[3] = state[2][self.ego_index]  # Lane number
        observation[4] = 1 if int(state[3][self.ego_index]) & 2 else 0 # Left indicator 
        observation[5] = 1 if int(state[3][self.ego_index]) & 1 else 0 # Right indicator 
        observation[6] = state[5][self.ego_index]  # Length of vehicle
        observation[7] = state[6][self.ego_index]  # Width of vehicle

        leader = traci.vehicle.getLeader(self.ego_id) # Returns leading vehicle id and distance
        if leader == None:
            leading_veh_dist = 1e6
        else:
            leading_veh_dist = leader[1]

        observation[8] = leading_veh_dist # Distance to leading vehicle

        s = self.nb_ego_states
        idx = 0
        for i, in_range in enumerate(vehicles_in_range):
            if not in_range:
                continue
            observation[s + idx * self.nb_states_per_vehicle] = (state[0][i, 0] - state[0][self.ego_index, 0]) # Longitudinal distance to ego vehicle
            observation[s + 1 + idx * self.nb_states_per_vehicle] = (state[0][i, 1] - state[0][self.ego_index, 1]) # Lateral distance to ego vehicle
            observation[s + 2  + idx * self.nb_states_per_vehicle] = (state[1][i, 0] - state[1][self.ego_index, 0]) # Relative longitudinal speed
            observation[s + 3  + idx * self.nb_states_per_vehicle] = np.sign(state[1][i, 1]) # Lane change state
            observation[s + 4  + idx * self.nb_states_per_vehicle] = state[2][i]  # Lane number
            observation[s + 5  + idx * self.nb_states_per_vehicle] = 1 if int(state[3][i]) & 2 else 0 # Left indicator 
            observation[s + 6  + idx * self.nb_states_per_vehicle] = 1 if int(state[3][i]) & 1 else 0 # Right indicator 
            observation[s + 7  + idx * self.nb_states_per_vehicle] = state[5][i]  # Length of vehicle
            observation[s + 8  + idx * self.nb_states_per_vehicle] = state[6][i]  # Width of vehicle

            idx += 1
            if idx >= self.sensor_nb_vehicles:
                break
        
        # If number of vehicles in range is less than sensor_nb_vehicles, fill with dummy values
        for i in range(idx, self.sensor_nb_vehicles):
            observation[s + idx * self.nb_states_per_vehicle] = 0
            observation[s + 1 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 2 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 3 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 4 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 5 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 6 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 7 + idx * self.nb_states_per_vehicle] = 0
            observation[s + 8 + idx * self.nb_states_per_vehicle] = 0
            idx += 1

        return observation

    def get_action_mask(self, obs):
        """
        Get action masks from all environments in a vectorized env.
        """
        traci.switch(self.env_label)
        mask = np.ones(self.action_space.n, dtype=np.float32)
        if (obs[3] == (self.nb_lanes - 1)):
            mask[CHANGE_LANE_LEFT] = 0
        elif (self.lat_controller.is_lanechange_safe(obs, self.nb_ego_states, self.nb_states_per_vehicle, obs[3] + 1) is False):
            mask[CHANGE_LANE_LEFT] = 0

        if (obs[3] == 0):
            mask[CHANGE_LANE_RIGHT] = 0
        elif (self.lat_controller.is_lanechange_safe(obs, self.nb_ego_states, self.nb_states_per_vehicle, obs[3] - 1) is False):
            mask[CHANGE_LANE_RIGHT] = 0
           
        return mask
    
    def print_info_in_gui(self):
        """
        Prints information in the GUI.
        """
        polygons = traci.polygon.getIDList()
        for polygon in polygons:
            traci.polygon.remove(polygon)
        dy = 20

        poly_ids_param = []
        poly_ids_param.append('Traffic Density: {0:.2f} vehicles per km'.format(round(self.traffic_density * 1000))) # Convert from veh/m to veh/km
        poly_ids_param.append('Moving Window Size: {0} m'.format(self.moving_window_size))
        poly_ids_param.append('HVR-Car: {0:.1f}, HVR-Trucks: {1:.1f}'.format(self.hvr_cars, self.hvr_trucks))
        poly_ids_param.append('-----------------------------------')

        poly_ids_info = []
        poly_ids_info.append('Ego Position: {0:.1f}, {1:.1f}'.format(self.positions[self.ego_index, 0] - self.init_ego_position, self.positions[self.ego_index, 1]))
        poly_ids_info.append('Ego Speed: {0:.1f}, {1:.1f}'.format(*self.speeds[self.ego_index, :]))

        #if len(self.kinetic_energy) > 0:
        #    poly_ids_info.append('Traffic flow kinetic energy sum: {0:.2f} kwh, Traffic flow kinetic energy sum error: {1:.2f} kwh'.format(self.kinetic_energy[self.step_], self.kinetic_energy[self.step_] - self.kinetic_energy[0]))
            #poly_ids_info.append('Traffic flow momentum: {0:.2f}'.format(self.momentum[self.step_]))

        for i in range(len(poly_ids_param)):
            traci.polygon.add(poly_ids_param[i], [self.positions[self.ego_index] + self.road.road_params['info_pos'], 
                                            self.positions[self.ego_index] + self.road.road_params['info_pos'] + [1, i * -dy]], 
                                            [1, 0, 0, 1])

            traci.polygon.addDynamics(poly_ids_param[i], self.ego_id)

        i = len(poly_ids_param)

        for j in range(len(poly_ids_info)):
            traci.polygon.add(poly_ids_info[j], [self.positions[self.ego_index] + self.road.road_params['info_pos'], 
                                            self.positions[self.ego_index] + self.road.road_params['info_pos'] + [1, -1 * ((i * dy) + (j * dy))]], 
                                            [0, 0, 0, 0])

            traci.polygon.addDynamics(poly_ids_info[j], self.ego_id)

    def close(self):
        super(Highway, self).close()
        traci.switch(self.env_label)        
        traci.close()
