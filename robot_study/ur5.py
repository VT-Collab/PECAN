import os 
import numpy as np
import pybullet as p
import pybullet_data

class UR5():

    def __init__(self, basePosition=[0, 0, 0]):

        self.urdfRootPath = "assets/robot/ur5.urdf"
        self.urdfAbsPath = os.path.normpath(os.path.join(os.path.dirname(__file__), self.urdfRootPath))
        self.ur5 = p.loadURDF(self.urdfAbsPath, useFixedBase=True, basePosition=basePosition)
        
        self.robotJoints = p.getNumJoints(self.ur5)
        self.ee_idx = self.robotJoints - 1

    def read_state(self):
        
        joint_states = p.getJointStates(self.ur5, range(self.robotJoints))
        link_states = p.getLinkStates(self.ur5, range(self.robotJoints))
        
        joint_position = np.zeros(self.robotJoints)
        joint_velocity = np.zeros(self.robotJoints)
        joint_torque = np.zeros(self.robotJoints)
        link_position = []
        link_quaternion = []

        for idx in range(self.robotJoints):
            joint_position[idx] = joint_states[idx][0]
            joint_velocity[idx] = joint_states[idx][1]
            joint_torque[idx] = joint_states[idx][3]
            link_position.append(list(link_states[idx][0]))
            link_quaternion.append(list(link_states[idx][1]))

        ee_states = p.getLinkState(self.ur5, self.ee_idx)
        ee_position = list(ee_states[4])
        ee_quaternion = list(ee_states[5])


        self.state['joint_position'] = joint_position
        self.state['joint_velocity'] = joint_velocity
        self.state['joint_torque'] = joint_torque
        self.state['ee_position'] = np.asarray(ee_position)
        self.state['ee_quaternion'] = np.asarray(ee_quaternion)
        self.state['link_position'] = link_position
        self.state['link_quaternion'] = link_quaternion

    def read_jacobian(self):
        linear_jacobian, angular_jacobian = p.calculateJacobian(self.ur5, self.ee_idx, [0, 0, 0], list(self.state['joint_position']), [0]*self.robotJoints, [0]*self.robotJoints)

        linear_jacobian = np.asarray(linear_jacobian)[:,:self.robotJoints]
        angular_jacobian = np.asarray(angular_jacobian)[:,:self.robotJoints]
        full_jacobian = np.zeros((6,self.robotJoints))
        full_jacobian[0:3,:] = linear_jacobian
        full_jacobian[3:6,:] = angular_jacobian 

        self.jacobian['full_jacobian'] = full_jacobian
        self.jacobian['linear_jacobian'] = linear_jacobian
        self.jacobian['angular_jacobian'] = angular_jacobian
        
        jacobians = []
        for i in range(self.robotJoints):
            linear_jacobian, angular_jacobian = p.calculateJacobian(self.ur5, self.ee_idx, [0, 0, 0], list(self.state['joint_position']), [0]*self.robotJoints, [0]*self.robotJoints)
            linear_jacobian = np.asarray(linear_jacobian)[:,:self.robotJoints]
            angular_jacobian = np.asarray(angular_jacobian)[:,:self.robotJoints]
            full_jacobian = np.zeros((6,self.robotJoints))
            full_jacobian[0:3,:] = linear_jacobian
            full_jacobian[3:6,:] = angular_jacobian
            jacobians.append(full_jacobian)

        self.jacobian['links_full_jacobian'] = jacobians


    def reset_robot(self, joint_position):

        self.state = {}
        self.jacobian = {}
        self.desired = {}

        for idx in range(len(joint_position)):
            p.resetJointState(self.ur5, idx, joint_position[idx])
        
        self.read_state()
        self.read_jacobian()

    def inverse_kinematics(self, ee_position, ee_quaternion):
        return p.calculateInverseKinematics(self.ur5, self.ee_idx, list(ee_position), list(ee_quaternion))


    def robot_control(self, mode='position', des_joint_pos=[0]*6, qdot=[0]*6):

        if mode == 'position':

            p.setJointMotorControlArray(self.ur5, range(self.robotJoints), p.POSITION_CONTROL, targetPositions=list(des_joint_pos))

        elif mode == 'velocity':
            p.setJointMotorControlArray(self.ur5, range(self.robotJoints), p.VELOCITY_CONTROL, targetVelocities=list(qdot))
