import aprel
import numpy as np
import pickle
from world import World
from agents import Car, RectangleBuilding, Painting, SpeedMeter, DistanceMeter
from geometry import Point
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import product
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()

class CARTrajectory:

    def __init__(self, env, trajectory):
        self.env = env
        self.trajectory = trajectory
        self.features = None
        self.speed = None
        self.distance = None

    def set_features(self, feature_values):
        self.features = feature_values

    def visualize(self):

        # create world
        dt = 0.2
        w = World(dt, width=120, height=120, ppm=6)

        if self.env == 'highway':

            # add buildings
            w.add(Painting(Point(103.5, 60), Point(35, 120), 'gray80'))
            w.add(RectangleBuilding(Point(104.5, 60), Point(32.5, 120)))
            w.add(Painting(Point(37.5, 60), Point(75, 120), 'gray80'))
            w.add(RectangleBuilding(Point(36.5, 60), Point(72.5, 120)))

            # add cars
            c1 = Car(Point(80, 20), np.pi / 2, 'orange')
            w.add(c1)
            c2 = Car(Point(80, 70), np.pi / 2, 'blue')
            w.add(c2)

        elif self.env == 'intersection':

            # add buildings
            w.add(Painting(Point(93.5, 106.5), Point(55, 27), 'gray80'))
            w.add(RectangleBuilding(Point(94.5, 107.5), Point(52.5, 25)))
            w.add(Painting(Point(27.5, 106.5), Point(55, 27), 'gray80'))
            w.add(RectangleBuilding(Point(26.5, 107.5), Point(52.5, 25)))
            w.add(Painting(Point(93.5, 41), Point(55, 82), 'gray80'))
            w.add(RectangleBuilding(Point(94.5, 40), Point(52.5, 80)))
            w.add(Painting(Point(27.5, 41), Point(55, 82), 'gray80'))
            w.add(RectangleBuilding(Point(26.5, 40), Point(52.5, 80)))

            # add crossings
            w.add(Painting(Point(56, 81), Point(0.5, 2), 'white'))
            w.add(Painting(Point(57, 81), Point(0.5, 2), 'white'))
            w.add(Painting(Point(58, 81), Point(0.5, 2), 'white'))
            w.add(Painting(Point(59, 81), Point(0.5, 2), 'white'))
            w.add(Painting(Point(60, 81), Point(0.5, 2), 'white'))
            w.add(Painting(Point(61, 81), Point(0.5, 2), 'white'))
            w.add(Painting(Point(62, 81), Point(0.5, 2), 'white'))
            w.add(Painting(Point(63, 81), Point(0.5, 2), 'white'))
            w.add(Painting(Point(64, 81), Point(0.5, 2), 'white'))
            w.add(Painting(Point(65, 81), Point(0.5, 2), 'white'))

            w.add(Painting(Point(67, 83), Point(2, 0.5), 'white'))
            w.add(Painting(Point(67, 84), Point(2, 0.5), 'white'))
            w.add(Painting(Point(67, 85), Point(2, 0.5), 'white'))
            w.add(Painting(Point(67, 86), Point(2, 0.5), 'white'))
            w.add(Painting(Point(67, 87), Point(2, 0.5), 'white'))
            w.add(Painting(Point(67, 88), Point(2, 0.5), 'white'))
            w.add(Painting(Point(67, 89), Point(2, 0.5), 'white'))
            w.add(Painting(Point(67, 90), Point(2, 0.5), 'white'))
            w.add(Painting(Point(67, 91), Point(2, 0.5), 'white'))
            w.add(Painting(Point(67, 92), Point(2, 0.5), 'white'))

            # add cars
            c1 = Car(Point(60, 35), np.pi / 2, 'orange')
            w.add(c1)
            c2 = Car(Point(80, 90), np.pi, 'blue')
            w.add(c2)

        else:
            raise NotImplementedError("Incorrect task specified.")

        # initialize sensors
        init_speed = 0
        sm = SpeedMeter(Point(25, 70), np.pi / 2, 'Max. Speed: ' + str(init_speed), 'green')
        w.add(sm)
        init_distance = np.round(c1.distanceTo(c2))
        dm = DistanceMeter(Point(26, 55), np.pi / 2, 'Min. Distance: ' + str(init_distance), 'red')
        w.add(dm)

        w.render()

        # initialise
        step = 0
        h_goal_y = 100.
        max_delta = 0.
        min_dist = np.inf

        # simulate
        while step < len(self.trajectory) and c1.center.y < h_goal_y:

            state = self.trajectory[step]

            delta_state = np.linalg.norm([state[0] - c1.center.x, state[1] - c1.center.y])

            if delta_state > max_delta:
                max_delta = deepcopy(delta_state)

            if c1.distanceTo(c2) < min_dist:
                min_dist = deepcopy(c1.distanceTo(c2))

            c1.center.x = state[0]
            c1.center.y = state[1]
            c2.center.x = state[2]
            c2.center.y = state[3]

            for agent in w.agents:
                if isinstance(agent, SpeedMeter):
                    delta_scale = 7.8  # 5.5  # 200
                    speed = np.round(max_delta*delta_scale)
                    agent.text = "Max. Speed: " + str(speed)
                if isinstance(agent, DistanceMeter):
                    distance = np.round(min_dist)
                    agent.text = "Min. Distance: " + str(distance)

            w.tick()
            w.render()
            time.sleep(dt)

            step += 1

        w.close()

        self.speed = speed
        self.distance = distance



def feature_func(traj):
    """Returns the features of the given trajectory, i.e. \Phi(traj).

    Args:
        traj: List of states, e.g. [state0, state1, ...]

    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """

    # return (np.array([min_pos, max_pos, mean_speed]) - mean_vec) / std_vec
    traj_array = np.array(traj)
    car_distances = [np.linalg.norm(s[:2] - s[2:]) for s in traj_array]
    min_distance = min(car_distances)

    car_speeds = [np.linalg.norm(traj_array[i+1, :2] - si[:2]) for i, si in enumerate(traj_array[:-1, :])]
    max_speed = max(np.round(car_speeds, decimals=2))

    # print(np.array(sorted(np.round(car_speeds, decimals=2).tolist()))*7.8)

    return np.array([max_speed, min_distance])


def balance_data(data):
    max_demo_len = max([len(demo) for demo in data])
    balanced_data = []
    for demo in data:
        balanced_demo = demo + [demo[-1]]*(max_demo_len - len(demo))
        balanced_data.append(balanced_demo)

    return balanced_data


def resample_data(data, max_length):
    resampled_data = []
    for demo in data:
        demo_idx = np.linspace(0, len(demo)-1, num=max_length, dtype=int)
        resampled_demo = np.array(demo)[demo_idx]
        resampled_data.append(resampled_demo.tolist())

    return resampled_data


def sample_weights(num_samples):
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    x = np.cos(angles)[:, np.newaxis]
    y = np.sin(angles)[:, np.newaxis]
    return np.hstack((x, y))


def checkTargetStyle(target_style, target_idx, save_Data=False):

    # Initialize optimizer
    trajectory_set = aprel.TrajectorySet(highway_trajectory_list)
    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    # Initialize real user
    true_user = aprel.HumanUser(delay=0.5)

    # Learn a reward function that is linear in trajectory features by assuming a softmax human response model
    features_dim = len(trajectory_set[0].features)
    params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
    user_model = aprel.SoftmaxUser(params)            

    belief = aprel.SamplingBasedBelief(user_model, [], params)
    # print('Estimated user parameters: ' + str(belief.mean))

    query = aprel.PreferenceQuery(trajectory_set[:2])

    # Initialize variables
    met_target = False

    # Initialize data to be saved
    num_queries = 0
    
    style_type = 'Practice'
    if save_Data:
        style_type = 'Teaching'
        final_weights_list = []
        final_traj_list = {'highway': [], 'intersection': []}
        rollout_style_list = {'highway': [], 'intersection': []}


    while not np.all(met_target):
        num_queries += 1

        if num_queries > 10:
            print(Fore.LIGHTYELLOW_EX + '[IMPORTANT] Maximum number of attempts exceed. Moving to next target' + Style.RESET_ALL)
            break
        
        print(Fore.LIGHTYELLOW_EX + '')
        print(style_type + " style", str(target_idx+1), ':', target_style)
        print('Target speed between', target_style[0] - style_tolerance[0], '-', target_style[0] + style_tolerance[0])
        print()
        print('Target distance between', target_style[1] - style_tolerance[1], '-', target_style[1] + style_tolerance[1], Style.RESET_ALL)        
        print('Query number: ', num_queries)
        print("Wait for the interface to show you a trajectory ...")

        queries, objective_values = query_optimizer.optimize('mutual_information', belief, query)
        # print('Objective Value: ' + str(objective_values[0]))

        responses = true_user.respond(queries[0])
        belief.update(aprel.Preference(queries[0], responses[0]))
        # print('Estimated user parameters: ' + str(belief.mean))

        # Select final trajectory for both environments
        final_weights = belief.mean['weights']
        highway_final_traj = selectFinalTraj(highway_trajectory_list, final_weights)
        intersection_final_traj = selectFinalTraj(intersection_trajectory_list, final_weights)

        # Visualize final trajectory
        print("This is what the autonomous car has learned to do:")
        time.sleep(2)
        highway_final_traj.visualize()
        intersection_final_traj.visualize()

        # Check if within tolerance
        rollout_style = [[highway_final_traj.speed, highway_final_traj.distance], [intersection_final_traj.speed, intersection_final_traj.distance]]
        rollout_style = np.round(rollout_style, 0).tolist()

        style_diff = np.abs(rollout_style - target_style)
        met_target = np.where(style_diff <= style_tolerance, True, False)

        # Print driving styles
        print()
        print(Fore.GREEN + '[INFO] Learned Style:', rollout_style)

        # Print helper message if target not met
        for task_idx, task_style in enumerate(rollout_style):

            if task_style[0] > target_style[0] + style_tolerance[0]:
                print('[INFO] Task {}: Speed too high'.format(task_idx + 1))

            elif task_style[0] < target_style[0] - style_tolerance[0]:
                print('[INFO] Task {}: Speed too low'.format(task_idx + 1))            

            
            if task_style[1] > target_style[1] + style_tolerance[1]:
                print('[INFO] Task {}: Distance too far'.format(task_idx + 1))

            elif task_style[1] < target_style[1] - style_tolerance[1]:
                print('[INFO] Task {}: Distance too close'.format(task_idx + 1))
        print(Style.RESET_ALL)    

        # Ask user if the want to continue practicing
        if not np.all(met_target) and not save_Data:
            stop = input("Do you want to stop practicing this style? (y/n): ")
            if stop == 'y':
                break
        
        if save_Data:
            # Update data
            final_weights_list.append(final_weights)
            final_traj_list['highway'].append(highway_final_traj.trajectory)
            final_traj_list['intersection'].append(intersection_final_traj.trajectory)
            rollout_style_list['highway'].append(rollout_style[0])
            rollout_style_list['intersection'].append(rollout_style[1])

            # Save data
            pickle.dump(num_queries, open(data_folder + '/baseline_trials'
                                          + str(target_idx+1) + '.pkl', 'wb'))
            pickle.dump(final_weights_list, open(data_folder + '/baseline_weights'
                                                + str(target_idx+1) + '.pkl', 'wb'))
            pickle.dump(final_traj_list, open(data_folder + '/baseline_trajectory' 
                                              + str(target_idx+1) + '.pkl', 'wb'))
            pickle.dump(rollout_style_list, open(data_folder + '/baseline_styles'
                                                 + str(target_idx+1) + '.pkl', 'wb'))
            

def selectFinalTraj(trajectory_list, final_weights):
    traj_features = np.array([traj.features for traj in trajectory_list])
    traj_rewards = np.dot(traj_features, final_weights.T)
    final_traj = trajectory_list[np.argmax(traj_rewards)]

    return final_traj


def generateTrajList(car_trajectories, environment):
    trajectory_list = []
    for ct in car_trajectories:
        traj = CARTrajectory(environment, ct)
        traj.set_features(feature_func(ct))
        trajectory_list.append(traj)

    return trajectory_list


def normalizeTrajFeatures(trajectory_list):
    features = np.array([traj.features for traj in trajectory_list])
    mean_features = np.mean(features, axis=0)
    std_features = np.std(features, axis=0)

    for traj in trajectory_list:
        scale_features = (traj.features - mean_features) / std_features
        traj.features = scale_features / np.linalg.norm(scale_features)

    return trajectory_list


    
h_vel_list = np.linspace(40, 100, 11)
h_dy_list = np.linspace(10, 30, 11)
h_styles = [h_style for h_style in product(h_vel_list, h_dy_list)]

# user id
ui = input("Enter user id: ")

# instruction
print("")
print("READ THE FOLLOWING INSTRUCTIONS:")
print("The interface will show you two options of car behavior.")
print("You have to select the one that is closest to your preference.")
print("Based on the option you select, the interface will try to learn your preferred speed and distance.")
print("Then, the interface will show you the behavior it has learned.")
print("You want the learned behavior (not the options) to match your preference.")

# load trajectories
highway_trajectories = pickle.load(open("data/highway_trajectories.pkl", "rb"))
intersection_trajectories = pickle.load(open("data/intersection_trajectories.pkl", "rb"))

# preprocess trajectories
trajectory_data = highway_trajectories + intersection_trajectories
trajectory_data = balance_data(trajectory_data)
max_len = min([30, min([len(demo) for demo in trajectory_data])])
trajectory_data_resample = resample_data(trajectory_data, max_len)
num_traj = int(len(trajectory_data_resample)/2)

# Simulation list
environments = ['highway', 'intersection']


# obtain trajectory options
start_idx = 0
end_idx = num_traj
highway_car_trajectories = np.array(trajectory_data_resample)[start_idx:end_idx]

start_idx = num_traj
end_idx = num_traj*2
intersection_car_trajectories = np.array(trajectory_data_resample)[start_idx:end_idx]

highway_trajectory_list = generateTrajList(highway_car_trajectories, environments[0])
intersection_trajectory_list = generateTrajList(intersection_car_trajectories, environments[1])

# Normalize features
highway_trajectory_list = normalizeTrajFeatures(highway_trajectory_list)
intersection_trajectory_list = normalizeTrajFeatures(intersection_trajectory_list)

# user styles
data_folder = 'carlo_study/user_' + ui
user_targets = pickle.load(open(data_folder + '/driving_styles.pkl', 'rb'))
# user_targets = user_targets[2:]
user_practice = pickle.load(open(data_folder + '/practice_driving_styles.pkl', 'rb'))

n_styles = user_targets.shape[0]
style_tolerance = [15, 5]

print()
print("----------------------------------------")
print(Fore.LIGHTYELLOW_EX + "[IMPORTANT] Your goal is to find each style in as few attempts as possible" + Style.RESET_ALL)
input('Press Enter to start the practice ...')

# Practice round
for practice_idx, practice_style in enumerate(user_practice):
    print()
    print("----------------------------------------")
    stop = input('Do you want to stop practice? (y/n): ')
    if stop == 'y':
        break
    checkTargetStyle(practice_style, practice_idx)

print()
print("----------------------------------------")
print(Fore.LIGHTYELLOW_EX + "[IMPORTANT] Your goal is to find each style in as few attempts as possible" + Style.RESET_ALL)
input('Press Enter to start the actual experiment ...')

for target_idx, target_style in enumerate(user_targets):
    print()
    print("----------------------------------------")
    checkTargetStyle(target_style, target_idx, save_Data=True)

print("Done.")