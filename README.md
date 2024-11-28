# Reinforcement Learning in Controlling Quadrotor UAV Flight Actions

Implementation of paper - [Application of Reinforcement Learning in Controlling Quadrotor UAV Flight Actions](https://doi.org/10.3390/drones8110660)

This repository is divided into two main sections. The first section, "multirotor," enables users to operate a simulated drone environment using keyboard controls. It integrates YOLOv7 with TensorRT to recognize targets within the environment and employs logical decision-making to achieve autonomous drone navigation through target frames. The second section, "reinforcement_learning," focuses on training the drone to autonomously pass through target frames using three reinforcement learning models (DQN, A2C, PPO). This includes methods utilizing raw image input as well as those incorporating YOLO. Among these, the PPO model with raw image input demonstrated the best performance in target traversal. The project is designed for a Windows-based computing environment.

<p align="center">
<img src="https://github.com/shawn243498/Reinforcement-Learning-in-Controlling-Quadrotor-UAV-Flight-Actions/blob/main/RLDroneTest.gif" width="300"/>
</p>

### Libraries & Tools
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
- [OpenAI - Gym](https://github.com/openai/gym)
- [Microsoft AirSim](https://github.com/microsoft/AirSim)
- [Unreal Engine 4](https://www.unrealengine.com/en-US/)
- [YOLOv7](https://github.com/WongKinYiu/yolov7/)
- [TensorRT(for YOLO)](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/)

## Overview

The environment includes both training and testing phases.

- In the training environment, there are 10 target frames. During training, the drone is randomly initialized at the center of one of these frames. If the drone successfully passes through the next target frame, it continues flying; otherwise, it is reset to the center of a randomly selected target frame.

- The testing environment features 11 target frames, with their positions differing from those in the training environment. A total of 11 tests are conducted, starting sequentially from the center of the first target frame. The goal is to evaluate how many target frames the drone can traverse. If the drone collides or fails to pass through a frame, it respawns at the center of the next target frame.

### Observation Space

- The input state consists of images, including RGB images and those processed through YOLO detection. 
- The image resolution is 360x240.

### Action Space
- There are 9 discrete actions.

## Environment setup

### For multirotor and No YOLO RL

**1. Clone the repository**

```
git clone https://github.com/shawn243498/Reinforcement-Learning-in-Controlling-Quadrotor-UAV-Flight-Actions.git
```
**2. From Anaconda command prompt, create a new conda environment**

I recommend you to use [Anaconda ](https://www.anaconda.com/products/individual-d) to create a virtual environment.

```
conda create -n RL_drone python==3.10
```
**3. Install required libraries**

Inside the main directory of the repo

```
conda activate RL_drone
cd (your path)/Reinforcement-Learning-in-Controlling-Quadrotor-UAV-Flight-Actions

python.exe -m pip install setuptools==65.5.0 pip==21
pip install wheel==0.38.0
pip install -r requirements.txt
```
**4. Edit `settings.json`**

Content of the settings.json should be as below:

> The `setting.json` file is located at `Documents\AirSim` folder.

```json
{
  "SettingsVersion": 1.2,
  "LocalHostIp": "127.0.0.1",
  "SimMode": "Multirotor",
  "ClockSpeed": 1,
  "ViewMode": "SpringArmChase",
  "Vehicles": {
    "drone0": {
      "VehicleType": "SimpleFlight",
      "X": 0.0,
      "Y": 0.0,
      "Z": 0.0,
      "Yaw": -90.0
    }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 360,
        "Height": 240,
        "FOV_Degrees": 82.6
      }
    ]
  }
}
```

## How to run the training?

**1. Download the training environment**

Go to the [releases](https://github.com/shawn243498/Reinforcement-Learning-in-Controlling-Quadrotor-UAV-Flight-Actions/releases) and download `Env_train.zip`. After downloading completed, extract it.

**2. Now, you can open up environment's executable file and start the training**

So, inside the repository
```
cd reinforcemenet_learning
python ppo_drone.py
```
## How to run the pretrained model?
**1. Download the test environment and the weight**

Go to the [releases](https://github.com/shawn243498/Reinforcement-Learning-in-Controlling-Quadrotor-UAV-Flight-Actions/releases) and download `Env_test.zip` . After downloading completed, extract it.
The weight is `ppo_navigation_policy.zip`, after download you need to it into the reinforcement_learning file in this repository.

#️⃣ **2. Now, you can open up environment's executable file and run the trained model**

So, inside the repository
```
cd reinforcemenet_learning
python ppo_policy_run.py
```








