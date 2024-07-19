import os
import aprel
import numpy as np
import pickle
from world import World
from agents import Car, RectangleBuilding, Painting, SpeedMeter, DistanceMeter
from geometry import Point
import time
import matplotlib.pyplot as plt
from copy import deepcopy


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


def visualize_trajectory(env, trajectory):

    # create world
    dt = 0.5
    w = World(dt, width=120, height=120, ppm=6)

    if env == 'highway':

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

    elif env == 'intersection':

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
    max_delta = 0.
    min_dist = np.inf

    # simulate
    for state in trajectory:

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
                delta_scale = 8
                speed = np.round(max_delta*delta_scale)
                agent.text = "Max. Speed: " + str(speed)
            if isinstance(agent, DistanceMeter):
                distance = np.round(min_dist)
                agent.text = "Min. Distance: " + str(distance)

        w.tick()
        w.render()
        time.sleep(dt)

    w.close()


def main():

    # load trajectories
    highway_trajectories = pickle.load(open("data/highway_trajectories.pkl", "rb"))
    intersection_trajectories = pickle.load(open("data/intersection_trajectories.pkl", "rb"))

    # preprocess trajectories
    trajectory_data = highway_trajectories + intersection_trajectories
    trajectory_data = balance_data(trajectory_data)
    max_len = min([30, min([len(demo) for demo in trajectory_data])])
    trajectory_data_resample = resample_data(trajectory_data, max_len)
    num_traj = int(len(trajectory_data_resample)/2)

    # run algo
    environments = ['highway', 'intersection']
    for env_idx, env_name in enumerate(environments):

        # obtain trajectory options
        start_idx = num_traj*env_idx
        end_idx = num_traj*(env_idx+1)
        car_trajectories = np.array(trajectory_data_resample)[start_idx:end_idx]

        rand_idx = np.random.choice(list(range(len(car_trajectories))))

        visualize_trajectory(env_name, car_trajectories[rand_idx])

    print("Done visualizing.")

    # user id
    ui = input("Enter user id: ")

    # select user styles
    num_styles = 4

    user_speeds = np.random.choice(np.linspace(40, 100, 61), num_styles*2, False).reshape(-1, 1)
    user_distances = np.random.choice(np.linspace(10, 30, 21), num_styles*2, False).reshape(-1, 1)
    user_styles = np.hstack((user_speeds, user_distances))


    print('Let the user personalize the car to have:')
    for i, style in enumerate(user_styles[:num_styles, :]):
        print('Style #' + str(i+1))
        print('Speed = ', style[0], 'Distance =', style[1])
        print()

    print('Let the user practice to personalize the car to have:')
    for i, style in enumerate(user_styles[num_styles:, :]):
        print(style)
        print('Style #' + str(i+1))
        print('Speed = ', style[0], 'Distance =', style[1])
        print()

    # create user folder
    data_folder = 'carlo_study/user_' + ui
    os.makedirs(data_folder)

    pickle.dump(user_styles[:num_styles, :], open(data_folder + '/driving_styles.pkl', 'wb'))
    pickle.dump(user_styles[num_styles:, :], open(data_folder + '/practice_driving_styles.pkl', 'wb'))


if __name__ == '__main__':
    main()