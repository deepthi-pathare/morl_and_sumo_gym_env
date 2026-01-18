import traci

def check_for_collision(ego_id):
    ego_collision, ego_near_collision, other_veh_collision = False, False, False

    collisions = traci.simulation.getCollisions()

    if collisions is None:
        return ego_collision, ego_near_collision, other_veh_collision

    for c in collisions:
        collision = is_collision(c)
        if ego_id == c.collider:
            if collision:
                ego_collision = True
                #print(c.__dict__)
                break
            else:
                ego_near_collision = True
        else:
            if collision:
                other_veh_collision = True
    
    return ego_collision, ego_near_collision, other_veh_collision
            
def is_collision(c):
    collision = True

    collider_pos = traci.vehicle.getPosition(c.collider)[0]
    victim_pos = traci.vehicle.getPosition(c.victim)[0]
    
    # Refer https://sumo.dlr.de/docs/Simulation/Output/Collisions.html
    if c.type == "collision":
        front_veh_length = traci.vehicle.getLength(c.victim)
        long_dist = victim_pos - collider_pos
    elif c.type == "frontal":# Not likely to occur since we only have vehicles in one direction
        front_veh_length = traci.vehicle.getLength(c.collider) 
        long_dist = collider_pos - victim_pos
    else:
        return collision
    
    if long_dist - front_veh_length > 0:
        collision = False

    #if collision:
    #    print(c.type, c.collider, c.victim, traci.vehicle.getPosition(c.collider), traci.vehicle.getPosition(c.victim))

    return collision
