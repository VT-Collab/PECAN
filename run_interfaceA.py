from utils import plotGUI, CARTrajectory, target_check_message
import time
import torch
import pickle
from models import *
from copy import deepcopy
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()


def actions_to_states(data_action, start_state):
    input_dim = 4
    seq_len = 30
    traj_action = data_action
    action_seq = np.reshape(traj_action, (seq_len, input_dim))
    state_seq = [np.array(start_state)]

    for seq_idx, action in enumerate(action_seq):
        state_seq.append(state_seq[seq_idx] + action)

    return state_seq

# rollout a trajectory given a z_task and z_style
def rollout(z_task, z_style, start_state):

    zt = torch.FloatTensor(z_task).unsqueeze(0)
    zs = torch.FloatTensor([z_style])
    action_traj = model.decoder(zt, zs).detach().squeeze(0).numpy()

    state_traj = actions_to_states(action_traj, start_state)

    return state_traj


def showCorners(init_styles):
    corners = {'styles': init_styles}
    env = 'highway'
    task_idx = 0

    for corner in corners['styles']:
        print("")

        # Launch gui
        gui = plotGUI(corners, show_Demos=True, corner=corner)

        z_task = task_embedding[task_idx]
        z_style = corner

        state_traj = rollout(z_task, z_style, start_states[task_idx])

        # Simulate
        car_traj = CARTrajectory(env, state_traj)
        car_traj.visualize()

        speed, distance = car_traj.speed, car_traj.distance

        print('Point:', corner, 'Style:', [speed, distance])


def checkTargetStyle(prev_styles, target_style, target_idx, save_Data=False):
    # Initialize variables
    task_idx = 0
    met_target = False
    count_trials = 0

    if save_Data:
        # Initialize data to be saved
        z_style_list = []
        rollouts_list = {'highway': [], 'intersection': []}
        rollouts_style_list = {'highway': [], 'intersection': []}


    while not np.all(met_target):
        count_trials += 1

        if count_trials > 10:
            print(Fore.LIGHTYELLOW_EX + '[IMPORTANT] Maximum number of attempts exceed. Moving to next target' + Style.RESET_ALL)
            break

        # Check if user selected a style loop until they do
        user_sel = False
        terminate = False

        while not user_sel and not terminate:
            # Launch gui
            gui = plotGUI(prev_styles, target_style=target_style)

            z_style = gui.style
            terminate = gui.terminate

            if z_style is not None:
                user_sel = True

            if terminate and save_Data:
                terminate = False

        # Terminate the function for this style
        if terminate:
            break
        
        # Update previous style list
        prev_styles['styles'].append(z_style)

        # Rollout a trajectory according to current z parameters
        highway_traj = rollout(task_embedding[0], z_style, start_states[0])
        intersection_traj = rollout(task_embedding[1], z_style, start_states[1])

        # Simulate trajectories
        highway_sim = CARTrajectory(env='highway', trajectory=highway_traj)
        highway_sim.visualize()
        highway_speed, highway_distance = highway_sim.speed, highway_sim.distance

        intersec_sim = CARTrajectory(env='intersection', trajectory=intersection_traj)
        intersec_sim.visualize()
        intersec_speed, intersec_distance = intersec_sim.speed, intersec_sim.distance

        # Check if rollout style is within tolerance
        rollout_style = [[highway_speed, highway_distance], [intersec_speed, intersec_distance]]
        style_diff = np.abs(rollout_style - target_style)
        met_target = np.where(style_diff <= style_tolerance, True, False)

        # Print driving styles
        print()
        print(Fore.LIGHTYELLOW_EX + 'Attempt:', count_trials)
        print(Fore.GREEN + '[INFO] Point:', np.round(z_style, 2), 'Style:', rollout_style)

        # Print helper message if target not met 
        target_check_message(target_style, rollout_style, style_tolerance)           
        print(Style.RESET_ALL)

        if save_Data:
            # Update data file
            z_style_list.append(z_style)
            rollouts_list['highway'].append(highway_traj)
            rollouts_list['intersection'].append(intersection_traj)
            rollouts_style_list['highway'].append(rollout_style[0])
            rollouts_style_list['intersection'].append(rollout_style[1])

            # Save data
            pickle.dump(count_trials, open(data_folder + '/ours_trials' + str(target_idx+1) + '.pkl', 'wb'))
            pickle.dump(z_style_list, open(data_folder + '/ours_z_styles' + str(target_idx+1) + '.pkl', 'wb'))
            pickle.dump(rollouts_list, open(data_folder + '/ours_trajectories' + str(target_idx+1) + '.pkl', 'wb'))
            pickle.dump(rollouts_style_list, open(data_folder + '/ours_styles' + str(target_idx+1) + '.pkl', 'wb'))


# Get user id
ui = input("Enter user id: ")

# Load demos for each task and start state for each rollout
dataset = pickle.load(open("data/dataset.pkl", "rb"))
start_states = [dataset[0][0], dataset[-1][0]]

# Define task embedding index order
task_embedding = [[0., 1.], [1., 0.]]

# load trained model
torch.manual_seed(0)
model = PECAN()
model.load_state_dict(torch.load('data/model_24'))
model.eval()

# Load list of user target styles
data_folder = 'carlo_study/user_' + ui
user_targets = pickle.load(open(data_folder + '/driving_styles.pkl', 'rb'))
user_practice = pickle.load(open(data_folder + '/practice_driving_styles.pkl', 'rb'))

# Define style tolerance
style_tolerance = [15, 5]


def main():
    init_styles = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
    prev_styles = {'styles': deepcopy(init_styles)}

    # show initial
    print("")
    print("----------------------------------------")
    print("Press ESC to see the four styles ...")
    gui = plotGUI(prev_styles)
    showCorners(init_styles)

    print()
    print("----------------------------------------")
    print(Fore.LIGHTYELLOW_EX + "[IMPORTANT] Your goal is to find each style in as few attempts as possible" + Style.RESET_ALL)
    input('Press Enter to start the practice ...')

    # Practice time
    for practice_idx, practice_style in enumerate(user_practice):

        print()
        print("----------------------------------------")
        print(Fore.LIGHTYELLOW_EX + "Practice style", str(practice_idx + 1), ':', practice_style)
        print('Target speed between', practice_style[0] - style_tolerance[0], '-', practice_style[0] + style_tolerance[0])
        print()
        print('Target distance between', practice_style[1] - style_tolerance[1], '-', practice_style[1] + style_tolerance[1], Style.RESET_ALL)   
        print()

        checkTargetStyle(prev_styles, practice_style, practice_idx)


    print()
    print("----------------------------------------")
    print(Fore.LIGHTYELLOW_EX + "[IMPORTANT] Your goal is to find each style in as few attempts as possible" + Style.RESET_ALL)
    input('Press Enter to start the actual experiment ...')

    # Re initialize prev_styles
    prev_styles = {'styles': deepcopy(init_styles)}

    for target_idx, target_style in enumerate(user_targets):

        print()
        print("----------------------------------------")
        print(Fore.LIGHTYELLOW_EX +"Teaching style", str(target_idx + 1), ':', target_style)
        print('Target speed between', target_style[0] - style_tolerance[0], '-', target_style[0] + style_tolerance[0])
        print()
        print('Target distance between', target_style[1] - style_tolerance[1], '-', target_style[1] + style_tolerance[1])   
        print(Style.RESET_ALL)

        checkTargetStyle(prev_styles, target_style, target_idx, save_Data=True)
      
    print()
    print("Closing Interface, You met all the targets")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass