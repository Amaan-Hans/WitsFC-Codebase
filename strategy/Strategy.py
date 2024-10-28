import math
import numpy as np
from math_ops.Math_Ops import Math_Ops as M



class Strategy():
    def __init__(self, world):
        self.play_mode = world.play_mode
        self.robot_model = world.robot  
        self.my_head_pos_2d = self.robot_model.loc_head_position[:2]
        self.player_unum = self.robot_model.unum
        self.mypos = (world.teammates[self.player_unum-1].state_abs_pos[0],world.teammates[self.player_unum-1].state_abs_pos[1])
       
        self.side = 1
        if world.team_side_is_left:
            self.side = 0

        self.teammate_positions = [teammate.state_abs_pos[:2] if teammate.state_abs_pos is not None 
                                    else None
                                    for teammate in world.teammates
                                    ]
        
        self.opponent_positions = [opponent.state_abs_pos[:2] if opponent.state_abs_pos is not None 
                                    else None
                                    for opponent in world.opponents
                                    ]



        

        self.team_dist_to_ball = None
        self.team_dist_to_oppGoal = None
        self.opp_dist_to_ball = None

        self.prev_important_positions_and_values = None
        self.curr_important_positions_and_values = None
        self.point_preferences = None
        self.combined_threat_and_definedPositions = None


        self.my_ori = self.robot_model.imu_torso_orientation
        self.ball_2d = world.ball_abs_pos[:2]
        self.ball_vec = self.ball_2d - self.my_head_pos_2d
        self.ball_dir = M.vector_angle(self.ball_vec)
        self.ball_dist = np.linalg.norm(self.ball_vec)
        self.ball_sq_dist = self.ball_dist * self.ball_dist # for faster comparisons
        self.ball_speed = np.linalg.norm(world.get_ball_abs_vel(6)[:2])
        
        self.goal_dir = M.target_abs_angle(self.ball_2d,(15.05,0))

        self.PM_GROUP = world.play_mode_group
        
        self.slow_ball_pos = world.get_predicted_ball_pos(0.5) # predicted future 2D ball position when ball speed <= 0.5 m/s

        # list of squared distances between teammates (including self) and slow ball (sq distance is set to 1000 in some conditions)
        self.teammates_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2)  # squared distance between teammate and ball
                                  if p.state_last_update != 0 and (world.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen
                                  else 1000 # force large distance if teammate does not exist, or its state info is not recent (360 ms), or it has fallen
                                  for p in world.teammates ]

        # list of squared distances between opponents and slow ball (sq distance is set to 1000 in some conditions)
        self.opponents_ball_sq_dist = [np.sum((p.state_abs_pos[:2] - self.slow_ball_pos) ** 2)  # squared distance between teammate and ball
                                  if p.state_last_update != 0 and world.time_local_ms - p.state_last_update <= 360 and not p.state_fallen
                                  else 1000 # force large distance if opponent does not exist, or its state info is not recent (360 ms), or it has fallen
                                  for p in world.opponents ]

        self.min_teammate_ball_sq_dist = min(self.teammates_ball_sq_dist)
        self.min_teammate_ball_dist = math.sqrt(self.min_teammate_ball_sq_dist)   # distance between ball and closest teammate
        
        sorted_teammate_ball_sq_dist = sorted(self.teammates_ball_sq_dist)
        self.second_min_teammate_ball_sq_dist = sorted_teammate_ball_sq_dist[1]
        
        self.min_opponent_ball_dist = math.sqrt(min(self.opponents_ball_sq_dist)) # distance between ball and closest opponent

        self.active_player_unum = self.teammates_ball_sq_dist.index(self.min_teammate_ball_sq_dist) + 1

        self.my_desired_position = self.mypos
        self.my_desired_orientation = self.ball_dir
        self.owners = []
        self.your_next_line = {1: "None", #joseph jostar esque
                        2: "None",
                        3: "None",
                        4: "None",
                        5: "None",
                        6: "None",
                        7: "None",
                        8: "None",
                        9: "None",
                        10: "None",
                        11: "None"}
        
        
        # Sort the squared distances between opponents and the ball while keeping track of indices
        sorted_opponent_indices = sorted(range(len(self.opponents_ball_sq_dist)), key=lambda i: self.opponents_ball_sq_dist[i])

        # Retrieve the indices of the three closest opponents
        first_closest_idx = sorted_opponent_indices[0]
        second_closest_idx = sorted_opponent_indices[1]
        third_closest_idx = sorted_opponent_indices[2]

        # Store the positions of the three closest opponents as tuples
        self.oppfirst = tuple(self.opponent_positions[first_closest_idx]) if self.opponent_positions[first_closest_idx] is not None else None
        self.oppsecond = tuple(self.opponent_positions[second_closest_idx]) if self.opponent_positions[second_closest_idx] is not None else None
        self.oppthird = tuple(self.opponent_positions[third_closest_idx]) if self.opponent_positions[third_closest_idx] is not None else None 
    # Sort the squared distances between teammates and the ball while keeping track of indices
    
        sorted_teammate_indices = sorted(range(len(self.teammates_ball_sq_dist)), key=lambda i: self.teammates_ball_sq_dist[i])

        # Retrieve the indices of the three closest teammates
        first_closest_teammate_idx = sorted_teammate_indices[0]
        second_closest_teammate_idx = sorted_teammate_indices[1]
        third_closest_teammate_idx = sorted_teammate_indices[2]

        # Store the positions of the three closest teammates as tuples
        self.teammatefirst = tuple(self.teammate_positions[first_closest_teammate_idx]) if self.teammate_positions[first_closest_teammate_idx] is not None else None
        self.teammatesecond = tuple(self.teammate_positions[second_closest_teammate_idx]) if self.teammate_positions[second_closest_teammate_idx] is not None else None
        self.teammatethird = tuple(self.teammate_positions[third_closest_teammate_idx]) if self.teammate_positions[third_closest_teammate_idx] is not None else None

    
    
    
    
    #======================================================================================================
    
        self.goal_position = (15.5, 0.0)  # Center of the goal
        self.goal_tolerance = 0.25  # Allow a slight offset if obstacles are near the goal center
        
        # Potential Fields parameters
        self.goal_attraction_coefficient = 5.0
        self.obstacle_repulsion_coefficient = 100.0
        self.obstacle_effect_radius = 1.0
        self.damping_coefficient = 0.3  # Damping for smooth movement
        self.previous_force = np.zeros(2)  # Track previous movement direction

        self.formation_tiers = {"back": -10, "mid": 0, "forward": 10}  # Relative tiers based on field zones

        self.defensive_bound_x = -10  # Limit for defenders near own goal line
        self.goal_shoot_margin = 0.2

    def potential_fields_pathfinding(self, start_pos, goal_pos=None):
        if goal_pos is None:
            goal_pos = self.goal_position

        # Calculate attraction vector toward goal with decay factor
        attraction_force = self.calculate_attraction(start_pos, goal_pos)
        
        # Calculate cumulative repulsion from obstacles
        repulsion_force = self.calculate_repulsion(start_pos)

        # Apply boundary correction
        boundary_correction = self.apply_boundary_correction(start_pos)

        # Combine forces, including boundary correction, and apply damping
        resulting_force = attraction_force + repulsion_force + boundary_correction + self.central_field_correction(start_pos) - self.damping_coefficient * self.previous_force
        self.previous_force = resulting_force
        next_position = start_pos + resulting_force

        return (next_position[0], next_position[1])

    def calculate_attraction(self, current_pos, goal_pos):
        direction_to_goal = np.array(goal_pos) - np.array(current_pos)
        distance_to_goal = np.linalg.norm(direction_to_goal)
        decay_factor = 1 - np.exp(-0.1 * distance_to_goal)
        direction_to_goal = direction_to_goal / distance_to_goal if distance_to_goal > 0 else np.zeros_like(direction_to_goal)

        # Add a boost to attraction based on boundary proximity
        boundary_proximity_boost = 1.5 if np.linalg.norm(self.apply_boundary_correction(current_pos)) > 0.5 else 1.0
        attraction_multiplier = 1.0 if self.ball_dist > 0.5 else 1.5  # Boost when close to ball

        attraction_force = self.goal_attraction_coefficient * direction_to_goal * decay_factor * attraction_multiplier * boundary_proximity_boost
        return attraction_force

    def calculate_repulsion(self, current_pos):
        total_repulsion = np.zeros(2)
        for opponent_pos in self.opponent_positions:
            if opponent_pos is not None:
                direction_from_obstacle = np.array(current_pos) - np.array(opponent_pos)
                distance_to_obstacle = np.linalg.norm(direction_from_obstacle)

                if distance_to_obstacle < self.obstacle_effect_radius:
                    direction_from_obstacle /= distance_to_obstacle
                    variable_repulsion = (self.obstacle_repulsion_coefficient / (distance_to_obstacle ** 2 + 0.1)) \
                                         * (0.8 if self.ball_dist < 0.5 else 1.0)
                    repulsion_force = variable_repulsion * direction_from_obstacle
                    total_repulsion += repulsion_force

        return total_repulsion

    def apply_boundary_correction(self, current_pos):
        correction_force = np.zeros(2)

        # Define stronger correction force based on proximity to boundary
        if abs(current_pos[0]) > 13:  # x-axis boundary
            distance_to_boundary_x = 15 - abs(current_pos[0])
            correction_force[0] = -1.0 / (distance_to_boundary_x + 0.1) * np.sign(current_pos[0])

        if abs(current_pos[1]) > 8:  # y-axis boundary
            distance_to_boundary_y = 10 - abs(current_pos[1])
            correction_force[1] = -1.0 / (distance_to_boundary_y + 0.1) * np.sign(current_pos[1])

        return correction_force
    
    def central_field_correction(self, current_pos):
        center_of_field = np.array([0, 0])
        direction_to_center = center_of_field - np.array(current_pos)
        central_attraction_force = 0.05 * direction_to_center / np.linalg.norm(direction_to_center)
        return central_attraction_force
    
    #======================================================================================================
        
    def GenerateTeamToTargetDistanceArray(self, target, world):
        for teammate in world.teammates:
            pass
        

    def IsFormationReady(self, point_preferences):
        
        is_formation_ready = True
        for i in range(1, 12):
            if i != self.active_player_unum: 
                teammate_pos = self.teammate_positions[i-1]

                if not teammate_pos is None:
                    distance = np.sum((teammate_pos - point_preferences[i]) **2)
                    if(distance > 0.3):
                        is_formation_ready = False

        return is_formation_ready

    def GetDirectionRelativeToMyPositionAndTarget(self,target):
        target_vec = target - self.my_head_pos_2d
        target_dir = M.vector_angle(target_vec)

        return target_dir
    
    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2)) 
    
    def calculate_orientation(self, target_pos, my_pos):
        current_orientation = self.my_ori
        target_vec = np.array(target_pos) - np.array(my_pos)
        orientation_radians = np.arctan2(target_vec[1], target_vec[0])
        orientation_degrees = np.degrees(orientation_radians)
        relative_orientation = orientation_degrees
    
        return relative_orientation
    
    def are_points_collinear(self, position, ball_pos, goal, tolerance=0.45):
        x1, y1 = position
        x2, y2 = ball_pos
        x3, y3 = goal
    
        # Calculate slopes; handle division by zero by returning False if undefined
        try:
            slope1 = (y2 - y1) / (x2 - x1)
            slope2 = (y3 - y2) / (x3 - x2)
        except ZeroDivisionError:
        # If both slopes are vertical (undefined), check if x-coordinates are equal
            return abs(x1 - x2) < tolerance and abs(x2 - x3) < tolerance
    
        # Check if the difference between slopes is within the tolerance
        return abs(slope1 - slope2) < tolerance
    
    def point_in_direction(self, position, goal, distance=0.2):
        x1, y1 = position
        x2, y2 = goal

        # Calculate direction vector
        direction_x = x2 - x1   
        direction_y = y2 - y1

        # Calculate magnitude of the direction vector
        magnitude = math.sqrt(direction_x**2 + direction_y**2)

        # Normalize the direction vector
        unit_direction_x = direction_x / magnitude
        unit_direction_y = direction_y / magnitude

        # Scale the unit vector by the desired distance
        scaled_x = unit_direction_x * distance
        scaled_y = unit_direction_y * distance

        # Calculate new position
        new_position = (x1 + scaled_x, y1 + scaled_y)

        return new_position  

    def next_position_to_startat(self, target, ball, myposition, startat, buffer_distance=0.4, angle_threshold=78):
        # Convert all points to numpy arrays
        target = np.array(target)
        ball = np.array(ball)
        myposition = np.array(myposition)
        startat = np.array(startat)

        # Vectors for the two segments
        target_to_ball = ball - target
        ball_to_myposition = myposition - ball

        # Calculate the angle between the two vectors
        dot_product = np.dot(target_to_ball, ball_to_myposition)
        magnitude_product = np.linalg.norm(target_to_ball) * np.linalg.norm(ball_to_myposition)
        angle_degrees = np.degrees(np.arccos(dot_product / magnitude_product))

        if abs(angle_degrees) < angle_threshold:
            # Calculate adjacent points to avoid the ball
            # Find perpendicular direction to target -> ball vector
            perpendicular_direction = np.array([-target_to_ball[1], target_to_ball[0]]) / np.linalg.norm(target_to_ball)
            
            # Left and right points around the ball
            left_of_ball = ball + perpendicular_direction * buffer_distance
            right_of_ball = ball - perpendicular_direction * buffer_distance

            # Determine the closest point to myposition among left, right, and startat
            distances = {
                "left": np.linalg.norm(myposition - left_of_ball),
                "right": np.linalg.norm(myposition - right_of_ball),
                "startat": np.linalg.norm(myposition - startat)
            }
            closest_point = min(distances, key=distances.get)

            if closest_point == "left":
                return tuple(left_of_ball)
            elif closest_point == "right":
                return tuple(right_of_ball)
            else:
                return tuple(startat)
        else:
            # If the angle is >= 80 degrees, move directly towards startat
            return tuple(startat)
    def get_player_ahead(self, position):
        ind = None
        distance = float('inf')  # Initialize to a very large number

        for i in range(11):
            if self.teammate_positions[i][0] > position[0] + 1.5:  # Check if player is ahead
                current_distance = self.distance(self.teammate_positions[i], position)
                if current_distance < distance:
                    ind = i
                    distance = current_distance

        # Return the closest teammate ahead or the position itself if none found
        return self.teammate_positions[ind] if ind is not None else position
