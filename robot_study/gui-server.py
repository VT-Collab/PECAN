import socket
import numpy as np
from tkinter import *
from utils import CLEAR_GUI 


def connect2simrobot():
    HOST = "127.0.0.1"
    PORT = 65432

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()

    return conn

def send2simrobot(conn, task_embedding, style_embedding, play_traj_flag, reset_env_flag, save_traj_flag):
    send_msg = "s," + str(task_embedding) + "," + str(style_embedding) + "," + str(int(play_traj_flag == True)) + "," + str(int(reset_env_flag == True)) + "," + str(int(save_traj_flag == True))
    conn.send(send_msg.encode())

def listen2simrobot(conn):
    state_length = 1 + 1 + 1 + 1
    data = conn.recv(1024).decode()
    state_str = list(data.split(","))

    for idx in range(len(state_str)):
        if state_str[idx] == "s":
            state_str = state_str[idx + 1: idx + 1 + state_length]
            break
    try:
        state_vector = [float(item) for item in state_str]
    except ValueError:
        return None
    
    if len(state_vector) is not state_length:
        return None
    
    state_vector = np.asarray(state_vector)
    state = {}
    state["traj_flag"] = state_vector[0]
    state["env_flag"] = state_vector[1]
    state['distance'] = state_vector[2]
    state['save_flag'] = state_vector[3]

    return state

def simrobotState(conn):
    while True:
        state = listen2simrobot(conn)
        if state is not None:
            break
    return state


print("[*] Awaiting connection from pybullet")
simrobot = connect2simrobot()
print("[*] Connected to pybullet simulation")

# Load GUI
root = Tk()
gui = CLEAR_GUI(root,3)

def main():
    while True:
        root.update() # update gui

        # Send current gui info to pybullet robot
        send2simrobot(simrobot, gui.task_embedding, gui.style_embedding, gui.play_traj_flag, gui.reset_env_flag, gui.save_traj_flag)

        # Read answer from pybullet client
        state = simrobotState(simrobot)
        gui.play_traj_flag = state['traj_flag']
        gui.reset_env_flag = state['env_flag']
        gui.save_traj_flag = state['save_flag']

        if state['distance'] == 0:
            gui.distance.set('--')

        else:
            gui.distance.set(str(round(state['distance'],2)) + ' cm')

if __name__ == '__main__':
    main()



