# PECAN: Personalizing Robot Behavior through a Learned Canonical Space

This repository provides our implementation of PECAN for the carlo environment, and the simulation interface used for the robot study (see video [here](https://youtu.be/wRJpyr23PKI)).


# Install
This repository has been tested on python 3.11

1. Clone this repository using 

```
git clone https://github.com/VT-Collab/PECAN.git
```

2. Install requirements with 
```
pip install -r requirements.txt
```

# Carlo Study
### Testing trained model

We have provided the trained model used in our user study.

1. Generate driving styles by running.
```
cd carlo_study
python get_driving_styles.py
```
This will show you an example simulation for both tasks (highway and intersection) and ask your for a user id (you can enter any number). It then generates 4 practice styles and 4 teaching styles (e.g. [speed, distance] = [95, 10]), under a new folder needed to run the study. 

2. Run PECAN's interface
```
python run_interfaceA.py
```

After running the script, you will be presentend with PECAN's interface, the terminal will prompt you with instructions to follow during the study. Note that it is possible to skip any practice styles by pressing 'escape' on the keyboard, this will not be the case for the teaching styles. 

### Re-training model

The training dataset consist 16 demonstrations. 8 labeled demonstrations corresponding to the extreme styles (4 extremes per task), and 8 unlabeled demonstrations (4 per task) sampled at random from the intermediate styles. The provided model can be re-trained from the following script: 

```
python train_model.py
```

You can then re-run the interface to interact with the new trained model.

# Robot Study

### Testing trained model

We have provided a trained model from the demonstrations provided by the experimenters for task1 and task2, and the extreme style demonstrations for task3 provided by one user.

1. Launch PECAN's interface by running
```
cd robot_study
python gui-server.py
```

2. On a separate terminal launch the Pybullet simulation
```
cd robot_study
python simur5-client.py
```

Similar to the previous study this script will prompt you with some options on the terminal. As before you can input any number for the user id

### Re-training model

In this study the dataset consist of 8 demonstrations. We provided 6 demonstrations for the first two tasks (2 extremes and 1 intermediate style per task), the remaining 2 demonstrations were provided by the user corresponding to the extreme style for the third task. The provideed model can be re-trained from the follow script:

```
python train_model.py
```


