# PECAN: Personalizing Robot Behavior through a Learned Canonical Space

This repository provides our implementation of PECAN for the carlo enviroment. 

# Install

Clone this repository using 

```
git clone https://github.com/VT-Collab/PECAN.git
```

# Testing trained model

We provide the trained model used in our user study (see video [here](https://youtu.be/wRJpyr23PKI)).

1. Generate driving styles by running.
```
python get_driving_styles.py
```
This will show you an example simulation for both tasks (highway and intersection) and ask your for a user id. It then generates 4 practice styles and 4 teaching styles (e.g. [speed, distance] = [95, 10]). It is possible to skip any practice styles by pressing 'escape' on the keyboard. 

2. Run PECAN's interface
```
python run_interfaceA.py
```

# Training model

The training dataset consist 24 demonstrations. 4 labeled demonstrations corresponding to the extreme styles and 8 unlabeled demonstrations per tasks. The provided model can be re-trained from the following script 

```
python train_model.py
```




