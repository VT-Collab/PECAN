import pybullet as p
import pybullet_data
from ur5 import UR5
from scipy.spatial.transform import Rotation
import numpy as np
import time
from utils import Trajectory
from objects import YCBObject
import os
from robotiq_85 import Robotiq85

class simpleEnv():

    def __init__(self, home_pos=None):

        # Set up simulation parameters
        physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(1.80, 144.40, -41.80, [0., 0., 0.])
        # Load Scene and objects
        tableRot = Rotation.from_euler('xyz', [0,0,90], degrees=True).as_quat()
        self.planeID = p.loadURDF("plane.urdf", basePosition=[0., 0., -0.625])
        self.tableID = p.loadURDF("table/table.urdf", basePosition=[0.1, 0., -0.625], baseOrientation=tableRot)

        # Initialize object info dictionary
        self.objectsInfo = {'Name': [], 'ID': [], 'Pos': [], 'Ori': []}

        # Load stand for the mug
        mugStand = YCBObject('004_sugar_box')
        mugStand.load()
        musgStandPos = [0.503, -0.582, 0.03]
        mugStandOri = Rotation.from_euler('xyz', [90, 90, 90], degrees=True).as_quat()
        self.addObjectInfo('Mug Stand', mugStand.body_id, musgStandPos, mugStandOri)
        p.resetBasePositionAndOrientation(mugStand.body_id, musgStandPos, mugStandOri)

        # Load YCB Mug
        mug = YCBObject('025_mug')
        mug.load()
        mugPos = [0.503, -0.582, 0.05+0.0381]
        mugOri = [0, 0, 0, 1]
        self.addObjectInfo('Mug', mug.body_id, mugPos, mugOri)
        p.resetBasePositionAndOrientation(mug.body_id, mugPos, mugOri)

        # Load plate holder
        holderPos = [0.483 -0.02, -0.434 + 0.015, 0.008]
        holderRot = Rotation.from_euler('xyz', [90,0,90], degrees=True).as_quat()
        plateHolder = p.loadURDF("assets/plate_holder/model.urdf", basePosition=holderPos, baseOrientation=holderRot, globalScaling=0.7)
        self.addObjectInfo('Plate Holder', plateHolder, holderPos, holderRot)

        # Load plate
        platePos = [0.473 -0.02, -0.515 + 0.02, 0.06 + 0.0481]
        plateRot = Rotation.from_euler('xyz', [90,0,0], degrees=True).as_quat()
        plate = p.loadURDF("assets/plate/model.urdf", basePosition=platePos, baseOrientation=plateRot, globalScaling=0.70)
        #info = p.getDynamicsInfo(plate, -1)
        p.changeDynamics(plate, -1, mass=0.01)
        self.addObjectInfo('Plate', plate, platePos, plateRot)

        # Load UR5 Robot
        self.ur5 = UR5(basePosition=[0., 0., 0.])

        # Set home position
        if home_pos is None:
            self.home_pos = [0.06237185, -0.94011067, -2.34906005, -1.42369371, 1.57071281, 0.06261311]
        else:
            self.home_pos = home_pos

        self.ur5.reset_robot(self.home_pos)

        # Load gripper
        gripper_path = "assets/robot/robotiq_85.urdf"
        self.gripper_ori = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_quat()
        gripper_pos = self.ur5.state['ee_position'] + [0., 0., -0.004]
        self.gripper = p.loadURDF(gripper_path, basePosition=gripper_pos, baseOrientation=self.gripper_ori)
        
        # Create a fixed joint between the robot and gripper
        joint_ori = Rotation.from_euler("xyz", [0, 0, -90], degrees=True).as_quat()
        p.createConstraint(self.ur5.ur5, self.ur5.ee_idx, self.gripper, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.024], [0, 0, 0.], parentFrameOrientation=joint_ori)

        self.gripper_control = Robotiq85(self.gripper)  # initialize gripper control


    def addObjectInfo(self, name, id, pos, ori):
        self.objectsInfo['Name'].append(name)
        self.objectsInfo['ID'].append(id)
        self.objectsInfo['Pos'].append(pos)
        self.objectsInfo['Ori'].append(ori)


    def reset_env(self):

        self.ur5.reset_robot(self.home_pos)
        p.resetBasePositionAndOrientation(self.gripper, self.ur5.state['ee_position'] + [0., 0., -0.0004], self.gripper_ori)
        self.gripper_control.reset_gripper([0]*self.gripper_control.numJoints)

        for i in range(len(self.objectsInfo['Name'])):
            p.resetBasePositionAndOrientation(self.objectsInfo['ID'][i], self.objectsInfo['Pos'][i], self.objectsInfo['Ori'][i])


        return self.ur5.state
    

    def reset_obj(self, obj_names):
        for name in obj_names:
            obj_idx = self.objectsInfo['Name'].index(name)
            p.resetBasePositionAndOrientation(self.objectsInfo['ID'][obj_idx], self.objectsInfo['Pos'][obj_idx], self.objectsInfo['Ori'][obj_idx])


    def close(self):
        p.disconnect()


    def stepSim(self):
        # Take simulation step
        p.stepSimulation()

        # Update states
        self.ur5.read_state()
        self.ur5.read_jacobian()   


    def showVisualSphere(self, position, radius, shape=p.GEOM_SPHERE, rgbaColor=[1,0,0,1]):
        arr_len = len(position)
        visualID = p.createVisualShapeArray(shapeTypes=[shape]*arr_len, radii=[radius]*arr_len, rgbaColors=[rgbaColor]*arr_len, visualFramePositions=position)
        sphereID = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visualID, basePosition=[0.,0.,0.])

        return visualID, sphereID
    
    
    def removeBody(self, bodyID):
        p.removeBody(bodyID)


    def quatDiff(self, q_curr, q_goal):
        q_curr_inv = Rotation.from_quat(q_curr).inv()
        q_goal = Rotation.from_quat(q_goal)
        q_diff = q_goal * q_curr_inv
        return q_diff.as_quat()
    
    # Convert xdot to qdot using the geomtric jacobian
    def xdot2qdot(self, jacobian, xdot):
        J = jacobian
        J_inv = np.linalg.pinv(J)
        return J_inv.dot(xdot)   

    def robotAction(self, goal, cur_pose, action_scale=0.01, traj_following=False, robot_working=True, space='ee'):

            if space == 'ee':
                ## xyz error
                robot_error_xyz = goal[:3] - cur_pose[:3]

                ##orientation error
                q_curr = Rotation.from_euler('xyz', cur_pose[3:], degrees=False).as_quat()
                q_goal = Rotation.from_euler('xyz', goal[3:], degrees=False).as_quat()
                q_diff = self.quatDiff(q_curr, q_goal)
                q_diff = Rotation.from_quat(q_diff)
                robot_error_orient = q_diff.as_euler('xyz', degrees=False)

                ## robot error
                robot_error = np.concatenate((robot_error_xyz, robot_error_orient))

            elif space == 'joint':
                robot_error = goal - cur_pose

            if traj_following:
                robot_action = robot_error

            else:

                if robot_working:
                
                    if np.linalg.norm(robot_error) > 0.01:
                        robot_action = (robot_error / np.linalg.norm(robot_error)) * action_scale
                    else:                
                        robot_action =  (robot_error / np.linalg.norm(robot_error)) * action_scale
                    
                else:
                    robot_action = np.zeros((6,))

            return robot_action
    

    def go2pose(self, pose, baseline_ori=[0]*3, _xyz=False, space='ee', action_scale=0.01, gripper_pose=0.085):

        goal = np.copy(pose)
        if _xyz:
            goal = np.concatenate((goal, [0, 0, 0]))
            goal[-3:] += baseline_ori ## setting baseline gripper orientation
        
        shutdown = False
        run = True
        
        while not shutdown:
            
            if space == 'ee':
                curr_pos = self.ur5.state['ee_position']
                curr_quaternion = self.ur5.state['ee_quaternion']
                curr_euler = Rotation.from_quat(curr_quaternion).as_euler('xyz', degrees=False)
                curr_pose = np.concatenate((curr_pos, curr_euler))
                goal_pose = self.constraint2workspace(goal)

            elif space == 'joint':
                curr_pose = self.ur5.state['joint_position']
                goal_pose = goal

            if np.linalg.norm(goal_pose - curr_pose) < 0.01 and run:
                shutdown = True
                run = False

            action = self.robotAction(goal_pose, curr_pose, robot_working=run, space=space, action_scale=action_scale)

            if space == 'ee':
                action = self.xdot2qdot(self.ur5.jacobian['full_jacobian'], action)

            #print(action)
            self.ur5.robot_control(mode='velocity', qdot=action)
            self.gripper_control.move_gripper(gripper_pose)
            self.stepSim()

        return curr_pose
        

    def playTrajectory(self, waypoints, total_time, action_scale=5, space='ee'):
        

        traj = Trajectory(waypoints, total_time, space=space)

        run = True
        stop = False
        start_time = time.time()

        while run:

            curr_time = time.time() - start_time

            goal_pose = traj.get(curr_time)

            if space == 'ee':
                ee_pos = self.ur5.state['ee_position']
                ee_ori = self.ur5.state['ee_quaternion']
                ee_ori = Rotation.from_quat(ee_ori).as_euler('xyz', degrees=False)
                current_pose = np.concatenate((ee_pos, ee_ori))
            elif space == 'joint':
                current_pose = self.ur5.state['joint_position']
            
            action = self.robotAction(goal=goal_pose, cur_pose=current_pose, traj_following=True, space=space)
            if space == 'ee':
                action = self.xdot2qdot(jacobian=self.ur5.jacobian['full_jacobian'], xdot=action)
            
            self.ur5.robot_control(mode='velocity', qdot=action * action_scale)
            self.stepSim()

            if curr_time >= total_time or stop:
                run = False
        
        stop_vel = [0]*6
        self.ur5.robot_control(mode='velocity', qdot=stop_vel)
        self.stepSim()


    def setGripperPose(self, pose):
        robot_state = self.ur5.state['joint_position']

        for _ in range(100):
            self.ur5.robot_control(mode='position', des_joint_pos=robot_state)
            self.gripper_control.move_gripper(pose)
            self.stepSim()
        

    def constraint2workspace(self, waypoints):
        limit_x = [0.02480, 0.61838]
        limit_y = [-0.72145,0.72145]
        limit_z = [0.3, 0.693]
        assert limit_x[0]<limit_x[1] and limit_y[0] < limit_y[1] and limit_z[0]<limit_z[1],\
        "the limits for the workspace constraints are invalid (lower limits greater than higher limit)"
        
        waypoints = np.asarray(waypoints)
        
        if len(waypoints.shape) == 1:
            waypoints[0] = np.clip(waypoints[0], limit_x[0], limit_x[1])
            waypoints[1] = np.clip(waypoints[1], limit_y[0], limit_y[1])
            waypoints[2] = np.clip(waypoints[2], limit_z[0], limit_z[1])
        else:
            waypoints[:, 0] = np.clip(waypoints[:, 0], limit_x[0], limit_x[1])
            waypoints[:, 1] = np.clip(waypoints[:, 1], limit_y[0], limit_y[1])
            waypoints[:, 2] = np.clip(waypoints[:, 2], limit_z[0], limit_z[1])
        
        return waypoints


