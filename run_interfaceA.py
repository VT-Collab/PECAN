from utils import plotGUI
import time
import torch
import pickle
from models import *
from world import World
from agents import Car, RectangleBuilding, Painting, SpeedMeter, DistanceMeter
from geometry import Point
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
    # zs_zt = torch.cat((zs, zt), 1)
    # action_traj = model.traj_decoder(zs_zt).detach().squeeze(0).numpy()
    action_traj = model.decoder(zt, zs).detach().squeeze(0).numpy()

    state_traj = actions_to_states(action_traj, start_state)

    return state_traj


def simulate_highway(w, dt, traj):
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

    # initialize sensors
    init_speed = 0
    sm = SpeedMeter(Point(25, 100), np.pi / 2, 'Max. Speed: ' + str(init_speed), 'green')
    w.add(sm)
    init_distance = np.round(c1.distanceTo(c2))
    dm = DistanceMeter(Point(26, 85), np.pi / 2, 'Min. Distance: ' + str(init_distance), 'red')
    w.add(dm)

    w.render()

    # initialise
    step = 0
    h_goal_y = 100.
    max_delta = 0.
    min_dist = np.inf

    # simulate
    while step < len(traj) and c1.center.y < h_goal_y:

        state = traj[step]

        delta_state = np.linalg.norm([state[0] - c1.center.x, state[1] - c1.center.y])

        if delta_state > max_delta:
            max_delta = deepcopy(delta_state)

        if c1.distanceTo(c2) < min_dist:
            min_dist = deepcopy(c1.distanceTo(c2))

        c1.center.x = state[0]
        c1.center.y = state[1]
        c2.center.y = state[3]

        for agent in w.agents:
            if isinstance(agent, SpeedMeter):
                delta_scale = 7.8  # 5.3  # 3.5
                speed = np.round(max_delta*delta_scale)
                agent.text = "Max. Speed: " + str(speed)
            if isinstance(agent, DistanceMeter):
                distance = np.round(min_dist)
                agent.text = "Min. Distance: " + str(distance)

        w.tick()
        w.render()
        time.sleep(dt)

        step += 1

    return speed, distance


def simulate_intersection(w, dt, traj):
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
    while step < len(traj) and c1.center.y < h_goal_y:

        state = traj[step]

        delta_state = np.linalg.norm([state[0] - c1.center.x, state[1] - c1.center.y])

        if delta_state > max_delta:
            max_delta = deepcopy(delta_state)

        if c1.distanceTo(c2) < min_dist:
            min_dist = deepcopy(c1.distanceTo(c2))

        c1.center.x = state[0]
        c1.center.y = state[1]
        c2.center.x = state[2]

        for agent in w.agents:
            if isinstance(agent, SpeedMeter):
                delta_scale = 7.8  # 5.3  # 3.5
                speed = np.round(max_delta*delta_scale)
                agent.text = "Max. Speed: " + str(speed)
            if isinstance(agent, DistanceMeter):
                distance = np.round(min_dist)
                agent.text = "Min. Distance: " + str(distance)

        w.tick()
        w.render()
        time.sleep(dt)

        step += 1

    return speed, distance


def showCorners(init_styles):
    corners = {'styles': init_styles}
    task_idx = 0
    for corner in corners['styles']:
        print("")

        # Launch gui
        gui = plotGUI(task_idx, corners, show_Demos=True, corner=corner)

        z_task = task_embedding[task_idx]
        z_style = corner

        state_traj = rollout(z_task, z_style, start_states[task_idx])

        # Simulate
        dt = 0.2
        w = World(dt, width=120, height=120, ppm=6)

        if task_idx == 0:
            speed, distance = simulate_highway(w, dt, state_traj)
        elif task_idx == 1:
            speed, distance = simulate_intersection(w, dt, state_traj)
        else:
            print("Task not implemented.")
            speed, distance = None, None
        w.close()

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
            gui = plotGUI(task_idx, prev_styles, target_style=target_style)

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
        dt = 0.2
        w = World(dt, width=120, height=120, ppm=6)
        highway_speed, highway_distance = simulate_highway(w, dt, highway_traj)
        w.close()

        w = World(dt, width=120, height=120, ppm=6)
        inter_speed, inter_distance = simulate_intersection(w, dt, intersection_traj)
        w.close()

        # Check if rollout style is within tolerance
        rollout_style = [[highway_speed, highway_distance], [inter_speed, inter_distance]]
        style_diff = np.abs(rollout_style - target_style)
        met_target = np.where(style_diff <= style_tolerance, True, False)

        # Print driving styles
        print()
        print(Fore.LIGHTYELLOW_EX + 'Attempt:', count_trials)
        print(Fore.GREEN + '[INFO] Point:', np.round(z_style, 2), 'Style:', rollout_style)

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
model = MyModel()
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

    task_idx = 0
    prev_styles = {'styles': deepcopy(init_styles)}

    # show initial
    print("")
    print("----------------------------------------")
    print("Press ESC to see the four styles ...")
    gui = plotGUI(task_idx, prev_styles)
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