import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys
import numpy as np
from world import World
from agents import Car, RectangleBuilding, Painting, SpeedMeter, DistanceMeter
from geometry import Point
from copy import deepcopy
import time

class plotGUI():
    
    def __init__(self, prev_styles, show_Demos=False, corner=None, 
                 target_style=None):
        
        # Initialize variables
        self.prev_styles = prev_styles['styles']
        self.terminate = False
        self.style = None
        first_time = True

        # Initialize figure
        self.fig, self.ax = plt.subplot_mosaic([
                                                ['main', 'button']],
                                                width_ratios=[2.5, 1],
                                                layout='constrained'
                                               ) 
        self.fig.suptitle('Current Target Style: ' + str(target_style) + ", Press 'Enter' or button to play")  

        # Configure play button
        playButton = Button(self.ax['button'], 'Play')
        playButton.on_clicked(self.playButton)

        # Setup demo plot behavior
        if show_Demos:
            self.style = corner
            first_time = False
            timer = self.fig.canvas.new_timer(interval=1000)
            timer.add_callback(closeDemo())
            timer.start()
        
        # Initial figure draw
        self.updatePlot(first_time=first_time)

        # Disable default matplolib mouse and keyboard behavior
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.button_press_handler_id)

        # Enable custom mouse and keyboard behavior
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        
        plt.show()
    
        
    def on_press(self, event):
        sys.stdout.flush()

        if event.key == 'enter':
            plt.close()
        
        if event.key == 'escape':
            self.terminate = True
            plt.close()


    def on_mouse(self, event):
        sys.stdout.flush()
        
        # mouse button left click
        if event.button == 1 and event.inaxes == self.ax['main']:
            self.style = (event.xdata, event.ydata)
            # self.updatePlot(style=self.style)
            self.updatePlot()

    def updatePlot(self, first_time=False):
        self.ax['main'].clear()
        self.ax['main'].grid()
        self.ax['main'].set_xlim(-1.05, 1.05)
        self.ax['main'].set_ylim(-1.05, 1.05)

        for style in self.prev_styles:
            self.ax['main'].scatter(style[0], style[1], c='tab:grey')
            plt.draw()

        if not first_time:
            self.ax['main'].scatter(self.style[0], self.style[1], c='tab:orange')
            plt.draw()

    def playButton(self, event):
        plt.close()

class closeDemo(object):

    def __init__(self):
        self.first = True

    def __call__(self):
        if self.first:
            self.first = False
            return
        plt.close()


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


def target_check_message(target_style, rollout_style, style_tolerance):
    for task_idx, task_style in enumerate(rollout_style):

        if task_style[0] > target_style[0] + style_tolerance[0]:
            print('[INFO] Task {}: Speed too high'.format(task_idx + 1))

        elif task_style[0] < target_style[0] - style_tolerance[0]:
            print('[INFO] Task {}: Speed too low'.format(task_idx + 1))            

        
        if task_style[1] > target_style[1] + style_tolerance[1]:
            print('[INFO] Task {}: Distance too far'.format(task_idx + 1))

        elif task_style[1] < target_style[1] - style_tolerance[1]:
            print('[INFO] Task {}: Distance too close'.format(task_idx + 1))  

