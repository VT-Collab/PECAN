import aprel
import numpy as np
import pickle
from utils import CARTrajectory, target_check_message
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import product
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()


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
        target_check_message(target_style, rollout_style, style_tolerance)
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