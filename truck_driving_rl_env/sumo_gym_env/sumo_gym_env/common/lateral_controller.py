import traci
import math
import numpy as np
from sumo_gym_env.common.energy_calc import calculate_energy_consumed
from sumo_gym_env.common import collision

EGO_LATERAL_SPEED = 0.8 # m/s

class Lateral_Controller:
    def __init__(self, ego_id, nb_lanes, lane_width, sim_step_length):
        self.ego_id = ego_id
        self.nb_lanes = nb_lanes
        self.lane_width = lane_width
        self.sim_step_length = sim_step_length
        self.energy_consumed = 0

    def change_to_left_lane(self):
        """
            Change ego vehicle to left lane
        """             
        # 1e15 is the durarion: the lane will be chosen for the given amount of time (in s).
        # If vehicle is in leftmost lane, this method has no effect.
        traci.vehicle.changeLaneRelative(self.ego_id, 1, 1e15)
        return self.update_simulation()

    def change_to_right_lane(self):
        """
            Change ego vehicle to right lane
        """             
        traci.vehicle.changeLaneRelative(self.ego_id, -1, 1e15)
        return self.update_simulation()
    
    def update_simulation(self):
        """
            Update simulation
        """            
        ego_collision, ego_near_collision, other_veh_collision = False, False, False
        lc_duration = self.lane_width/EGO_LATERAL_SPEED
        lc_steps = math.ceil(lc_duration/self.sim_step_length)
        self.energy_consumed = 0.0
        
        for i in range(0, lc_steps):     
            traci.simulationStep()
 
            ec, enc, ovc = collision.check_for_collision(self.ego_id)
            
            slope = traci.vehicle.getSlope(self.ego_id)
            v_long = traci.vehicle.getSpeed(self.ego_id)
            v_total = math.sqrt(v_long**2 + EGO_LATERAL_SPEED**2)
            a_long = traci.vehicle.getAcceleration(self.ego_id)
            self.energy_consumed += calculate_energy_consumed(a_long, v_total, slope, self.sim_step_length)

            if ec:
                ego_collision = True
                break
            if enc:
                ego_near_collision = True
            if ovc:
                other_veh_collision = True

        return ego_collision, ego_near_collision, other_veh_collision

    def is_lanechange_safe(self, obs, nb_ego_states, nb_states_per_vehicle, target_lane):
        """
        Validates if a lane change to the target lane is safe.

        This function considers:
        - Vehicle lengths (positions are front bumper locations)
        - The ego vehicle occupies both lanes during lane change
        - Relative velocities for gap calculations
        - Safety in both current and target lanes during the maneuver

        Args:
            obs (ndarray): Observation of the environment
            nb_ego_states (int): Number of ego state variables
            nb_states_per_vehicle (int): Number of state variables per vehicle
            target_lane (int): Target lane ID to change to

        Returns:
            bool: True if lane change is safe, False otherwise
        """

        # Lane change parameters
        lc_duration = self.lane_width / EGO_LATERAL_SPEED  # 4 seconds

        # Safety parameters
        T_safe = 1.0  # Safe time gap in seconds
        b_safe = traci.vehicle.getDecel(self.ego_id)  # Comfortable braking limit (m/s²)
        a_max = traci.vehicle.getAccel(self.ego_id)  # Maximum acceleration limit (m/s²)
        s_0 = traci.vehicle.getMinGap(self.ego_id)  # Minimum standstill distance (m)

        # Get ego vehicle information
        v_ego = obs[1]  # Longitudinal speed
        ego_lane = obs[3]  # Current lane number
        ego_length = obs[6] # Ego vehicle length
        ego_width = obs[7] # Ego vehicle width

        # Calculate when ego vehicle enters and exits each lane
        # Ego vehicle starts entering target lane when it has moved laterally by (LANE_WIDTH - EGO_WIDTH)/2
        # and fully exits current lane when it has moved (LANE_WIDTH + EGO_WIDTH)/2
        t_enter_target = ((self.lane_width - ego_width) / 2) / EGO_LATERAL_SPEED  # ~0.4s
        t_exit_current = ((self.lane_width + ego_width) / 2) / EGO_LATERAL_SPEED  # ~3.6s

        # Check if target lane is the same as current lane
        if target_lane == ego_lane:
            return False  # Already in target lane

        # Initialize vehicle tracking dictionaries
        front_vehicle_cur = None
        rear_vehicle_cur = None
        front_vehicle_target = None
        rear_vehicle_target = None

        min_front_dist_cur = float('inf')
        min_rear_dist_cur = float('inf')
        min_front_dist_target = float('inf')
        min_rear_dist_target = float('inf')

        # Process all vehicle observations
        idx = nb_ego_states
        while idx < len(obs):
            long_dist = obs[idx]  # Distance from ego front bumper to other vehicle's front bumper
            lat_dist = obs[idx + 1]
            rel_speed = obs[idx + 2]
            vehicle_lane = obs[idx + 4]
            vehicle_length = obs[idx + 7]
            vehicle_width = obs[idx + 8]

            # Skip dummy vehicles (all zeros)
            if long_dist == 0 and lat_dist == 0 and rel_speed == 0 and vehicle_lane == 0:
                idx += nb_states_per_vehicle
                continue

            v_vehicle = v_ego + rel_speed

            # Vehicle in current lane
            if vehicle_lane == ego_lane:
                if long_dist > 0 and long_dist < min_front_dist_cur:
                    # Front bumper to front bumper distance
                    min_front_dist_cur = long_dist
                    front_vehicle_cur = {
                        's_front': long_dist,  # Front bumper to front bumper
                        'v_front': v_vehicle,
                        'length': vehicle_length
                    }
                elif long_dist < 0 and abs(long_dist) < min_rear_dist_cur:
                    # Vehicle behind in current lane
                    min_rear_dist_cur = abs(long_dist)
                    rear_vehicle_cur = {
                        's_rear': abs(long_dist),  # Front bumper distance (ego behind)
                        'v_rear': v_vehicle,
                        'length': vehicle_length
                    }

            # Vehicle in target lane
            elif vehicle_lane == target_lane:
                if long_dist > 0 and long_dist < min_front_dist_target:
                    # Vehicle ahead in target lane
                    min_front_dist_target = long_dist
                    front_vehicle_target = {
                        's_front': long_dist,
                        'v_front': v_vehicle,
                        'length': vehicle_length
                    }
                elif long_dist < 0 and abs(long_dist) < min_rear_dist_target:
                    # Vehicle behind in target lane
                    min_rear_dist_target = abs(long_dist)
                    rear_vehicle_target = {
                        's_rear': abs(long_dist),
                        'v_rear': v_vehicle,
                        'length': vehicle_length
                    }

            idx += nb_states_per_vehicle

        # Safety checks
        safe = True

        # 1. Check front gap in CURRENT lane (need to maintain safety while still partially in current lane)
        # Must maintain safe gap until we exit current lane (t_exit_current)
        if front_vehicle_cur is not None:
            v_rel = v_ego - front_vehicle_cur['v_front']  # Positive if we're catching up

            # Account for vehicle length: actual gap is front-to-front minus front vehicle length
            actual_gap = front_vehicle_cur['s_front'] - front_vehicle_cur['length']

            # Minimum safe gap based on velocity difference
            s_min_front_cur = s_0 + max(0, T_safe * v_ego + (v_ego * v_rel) / (2 * np.sqrt(a_max * b_safe)))

            # We need this gap for the duration we're still in current lane
            gap_at_exit = actual_gap - v_rel * t_exit_current

            if gap_at_exit < s_min_front_cur:
                safe = False

        # 2. Check front gap in TARGET lane
        # We enter target lane at t_enter_target and stay until end of maneuver
        if front_vehicle_target is not None:
            v_rel = v_ego - front_vehicle_target['v_front']

            # Account for vehicle length
            actual_gap = front_vehicle_target['s_front'] - front_vehicle_target['length']

            # Minimum safe gap
            s_min_front_target = s_0 + max(0, T_safe * v_ego + (v_ego * v_rel) / (2 * np.sqrt(a_max * b_safe)))

            # Check gap when we enter target lane
            gap_at_entry = actual_gap - v_rel * t_enter_target

            if gap_at_entry < s_min_front_target:
                safe = False

            # Also check gap at end of maneuver
            gap_at_end = actual_gap - v_rel * lc_duration
            if gap_at_end < s_min_front_target:
                safe = False

        # 3. Check rear gap in TARGET lane (most critical check)
        if rear_vehicle_target is not None:
            v_rel_rear = rear_vehicle_target['v_rear'] - v_ego  # Positive if rear is catching up

            # Actual gap: distance between our rear bumper and their front bumper
            actual_gap_rear = rear_vehicle_target['s_rear'] - ego_length

            # Minimum safe gap for rear vehicle (they need to maintain this)
            s_min_rear_target = s_0 + max(0, T_safe * rear_vehicle_target['v_rear'] + 
                                           (rear_vehicle_target['v_rear'] * v_rel_rear) / (2 * np.sqrt(a_max * b_safe)))

            # Check if gap is sufficient when we enter target lane
            gap_at_entry = actual_gap_rear + v_rel_rear * t_enter_target  # Gap decreases if rear is faster

            if gap_at_entry < s_min_rear_target:
                safe = False

            # Check if rear vehicle would need to brake too hard
            if rear_vehicle_target['v_rear'] > v_ego:
                # Time to collision if no braking
                ttc = actual_gap_rear / v_rel_rear if v_rel_rear > 0 else float('inf')

                # If collision would occur during lane change, check required braking
                if ttc < lc_duration:
                    required_decel = v_rel_rear / max(ttc - t_enter_target, 0.1)  # Avoid division by zero
                    if required_decel > b_safe:
                        safe = False

        return safe