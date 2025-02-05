import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tkinter import *
from tkinter import ttk
from scipy.interpolate import interp1d

def DH_matrix(theta, a, d, alpha):
    DH_mat = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                       [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                       [0, np.sin(alpha), np.cos(alpha), d],
                       [0, 0, 0, 1]])
    
    return DH_mat


def UR5_fk(theta, target_joint=6, twist_vec=True):
    # DH parameters : [a(m), d(m), alpha(rad)]
    UR5_DH_PARAM = np.array([[0, 0.089159, np.pi/2],
                            [-0.425, 0, 0],
                            [-0.39225, 0, 0],
                            [0, 0.10915, np.pi/2],
                            [0, 0.09465, -np.pi/2],
                            [0, 0.0823, 0]])

    T = np.identity(4)

    for i in range(target_joint):
        DH_mat = DH_matrix(theta[i], UR5_DH_PARAM[i,0], UR5_DH_PARAM[i,1], UR5_DH_PARAM[i,2])
        T = T.dot(DH_mat)

    if twist_vec:
        R = Rotation.from_matrix(T[:3, :3]).as_euler('xyz',degrees=False)
        p = T[:3, -1]
        T = np.concatenate((p, R))

    return T


class CLEAR_GUI:

    def __init__(self, root, n_task):

        root.title("Interface")
        font = 'Palatino Linotype'

        mainframe = ttk.Frame(root, padding="3 3 12 12", width=250, height=200)
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        buttonsFrame = ttk.Frame(root, padding="3 3 12 12", width=250, height=200)
        buttonsFrame.grid(column=3, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        # Set up GUI theme
        s = ttk.Style()
        s.theme_use('alt')

        # Configure task selection as listbox items
        self.task_embedding = 0
        task_sel = ["Task " + str(i+1) for i in range(n_task)]
        task_var = StringVar(value=task_sel)        
        self.task_list = Listbox(mainframe, listvariable=task_var)
        self.task_list.selection_set(0)
        self.task_list.grid(column=0, row=3, sticky='nwes')
        self.task_list.bind('<<ListboxSelect>>', self.updateTask)

        # Variables to store and display slider values
        self.style1 = DoubleVar(value=0.)
        self.show_style1 = StringVar(value=str(self.style1.get()))
        self.style_embedding = self.style1.get()

        # Configure slider 1
        s1 = ttk.Scale(mainframe, orient=VERTICAL, length=300, from_=1., to=-1., variable=self.style1, command=self.update_style)
        s1.grid(column=1, row=3, padx=10, pady=10, sticky='nwes', rowspan=3)
        ttk.Label(mainframe, text='Style 1').grid(row=2, column=1)
        ttk.Label(mainframe, textvariable=self.show_style1).grid(row=6, column=1)

        # Labels to show current distance to the human
        self.distance = StringVar(value='--')
        ttk.Label(mainframe, text='Distance to human').grid(column=2, row=2)
        ttk.Label(mainframe, textvariable=self.distance, font=(font, 40)).grid(column=2, row=3)

        # Configure button to play trajectory on the sim robot
        self.play_traj_flag = False
        self.sim_play = ttk.Button(buttonsFrame, text='Play Sim Trajectory', command=self.play_traj)
        self.sim_play.grid(column=0, row=2, padx=20, pady=10, sticky='nwes')

        # Configure button to reset pybullet env
        self.reset_env_flag = False
        reset = ttk.Button(buttonsFrame, text='Reset Simulation', command=self.reset_env)
        reset.grid(column=0, row=1, padx=20, pady=10, sticky='nwes')

        # Configure button to play trajectory on the real robot
        self.robot_traj_flag = False
        self.robot_play = ttk.Button(buttonsFrame, text='Play Trajectory on Robot', command=self.robot_traj)
        self.robot_play.grid(column=0, row=4, padx=20, pady=10, sticky='nwes')
        self.robot_play.state(['disabled'])

        # Configure button to save trajectory for real robot
        self.save_traj_flag = False
        self.save_traj = ttk.Button(buttonsFrame, text='Save Trajectory on Robot', command=self.save2robot)
        self.save_traj.grid(column=0, row=3, padx=20, pady=10, sticky='nwes')


        mainframe.columnconfigure(0, weight=1)
        mainframe.columnconfigure(1, weight=1)
        mainframe.columnconfigure(2, weight=1)
        mainframe.columnconfigure(3, weight=2)
        mainframe.columnconfigure(4, weight=2)
        mainframe.rowconfigure(1, weight=1)
        mainframe.rowconfigure(2, weight=1)
        mainframe.rowconfigure(3, weight=1)
    
    # Call back function that updates the current task selection
    def updateTask(self, *args):
        if self.task_list.curselection():
            self.task_embedding = self.task_list.curselection()[0]

        # NOTE: in the actual user study, user were prevented from playing the sim for task 3
        # if self.task_embedding < 2:
        #     self.sim_play.state(['!disabled'])

        # elif self.task_embedding >= 2:
        #     self.sim_play.state(['disabled'])
            
    # Callback function that updates the current slider values
    def update_style(self, *args):
        value1 = self.style1.get()
        self.show_style1.set(round(value1,2))
        self.style_embedding = value1

    # Callback function that updates the current radion button values
    def update_task(self, *args):
        task_sel = self.task_sel.get()

        if task_sel == 'task1':
            self.task_embedding = 0.  # [0., 1., 0.]

        elif task_sel == 'task2':
            self.task_embedding = 1.  # [1., 0., 0.]
        
        elif task_sel == 'task3':
            self.task_embedding = 2.  # [0., 0., 1.]


    # Callback function for the sim play trajectory button
    def play_traj(self, *args):
        self.play_traj_flag = not self.play_traj_flag


    # Callback function for the reset env button
    def reset_env(self, *args):
        self.reset_env_flag = not self.reset_env_flag

    
    # Callback function for the robot trajectory button
    def robot_traj(self, *args):
        self.robot_traj_flag = not self.robot_traj_flag
        self.robot_play.state(['disabled'])
        self.save_traj.state(['!disabled'])

    # Callback funtion for the save trajectory button
    def save2robot(self, *args):
        self.save_traj_flag = not self.save_traj_flag
        self.save_traj.state(['disabled'])
        self.robot_play.state(['!disabled'])

    
class Trajectory(object):

    def __init__(self, waypoints, total_time, kind='linear', space='joint'):
        # Create function interpolators between waypoints
        self.waypoints = np.asarray(waypoints)
        self.T = total_time

        self.n_waypoints = self.waypoints.shape[0]

        if self.n_waypoints <= 2:
            timesteps = np.linspace(0,self.T, self.n_waypoints)
        
        else:
            timesteps = self.assignTimesteps(self.waypoints, self.T)
        
        self.f1 = interp1d(timesteps, self.waypoints[:,0], kind=kind)
        self.f2 = interp1d(timesteps, self.waypoints[:,1], kind=kind)
        self.f3 = interp1d(timesteps, self.waypoints[:,2], kind=kind)

        self.space = space

        if self.space == 'joint':
            self.f4 = interp1d(timesteps, self.waypoints[:,3], kind=kind)
            self.f5 = interp1d(timesteps, self.waypoints[:,4], kind=kind)
            self.f6 = interp1d(timesteps, self.waypoints[:,5], kind=kind)

        elif self.space == 'ee':
            rotations = Rotation.from_euler('xyz', waypoints[:, 3:], degrees=False)
            self.rots = Slerp(timesteps, rotations)

    # Assign timestep for each waypoint based on normalized distance
    def assignTimesteps(self, waypoints, total_time):

        distance_array = np.zeros(waypoints.shape[0] - 1)

        for i, waypoint in enumerate(waypoints[:-1]):
            distance_array[i] = np.linalg.norm(waypoints[(i+1),:] - waypoint)
            
        norm_distance_arr = distance_array / np.sum(distance_array)
        timesteps = np.array([0])

        for norm_distance in norm_distance_arr:
            timesteps = np.append(timesteps, (timesteps[-1] + norm_distance * total_time))

        timesteps[-1] = total_time
        
        return timesteps


    def get(self, t):
        # get interpolated position
        if self.space == 'joint':
            q = self._get_joint(t)
        elif self.space == 'ee':
            q = self._get_ee(t)

        return q
    
    def _get_joint(self, t):
        # get interpolated position
        if t < 0:
            q = [self.f1(0), self.f2(0), self.f3(0), self.f4(0), self.f5(0), self.f6(0)]
        elif t < self.T:
            q = [self.f1(t), self.f2(t), self.f3(t), self.f4(t), self.f5(t), self.f6(t)]
        else:
            q = [self.f1(self.T), self.f2(self.T), self.f3(self.T), self.f4(self.T), self.f5(self.T), self.f6(self.T)]

        return np.asarray(q)
    
    def _get_ee(self, t):
        # get interpolated position
        if t < 0:
            interp_rots = self.rots([0]).as_euler('xyz', degrees=False)[0]
            q = [self.f1(0), self.f2(0), self.f3(0), interp_rots[0], interp_rots[1], interp_rots[2]]

        elif t < self.T:
            interp_rots = self.rots([t]).as_euler('xyz', degrees=False)[0]
            q = [self.f1(t), self.f2(t), self.f3(t), interp_rots[0], interp_rots[1], interp_rots[2]]
        else:
            interp_rots = self.rots([self.T]).as_euler('xyz', degrees=False)[0]
            q = [self.f1(self.T), self.f2(self.T), self.f3(self.T), interp_rots[0], interp_rots[1], interp_rots[2]]
        return np.asarray(q)        