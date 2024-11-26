import airsim
from queue import Queue
import threading
import cv2
import time
import sys
import numpy as np
import math
import yaml

############ Subroutine ############
from trtforpython import TRTInitial, Detect
import KeyPressModule as kp

def printUsage():
   print("Usage: Choosing wrong cameraType")

def getKeyboardInput():
    lr, fb, ud, yv, status = 0, 0, 0, 0, False
    speed1 = 5

    if kp.getKey("RIGHT"): lr = speed1
    elif kp.getKey("LEFT"): lr = -speed1

    if kp.getKey("UP"): fb = speed1
    elif kp.getKey("DOWN"): fb = -speed1

    if kp.getKey("w"): ud = -speed1
    elif kp.getKey("s"): ud = speed1

    if kp.getKey("a"): yv = -speed1*2
    elif kp.getKey("d"): yv = speed1*2

    # autopilot mode
    if kp.getKey("k"): status = True

    return [lr, fb, ud, yv, status]

def KeyboardControl(FlyAction):
    kp.init()
    while True:
        # Keyboard Control
        vals = getKeyboardInput()
        # put keyboard command to FlyAction Queue
        FlyAction.put(vals)

def DroneFly(FlyAction, DetectResult):
    # connect to drone in simulation
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    w = 360
    h = 240

    while True:
        # wait for FlyAction vals send
        vals = FlyAction.get()
        # wait for DetectResult vals send
        result = DetectResult.get()

        # if user not press "k", using manual control mode
        if vals[4] is not True:
            # airsim control command
            client.moveByVelocityBodyFrameAsync(vals[1], vals[0], vals[2], 0.02,
                                          yaw_mode=airsim.YawMode(True, yaw_or_rate=vals[3]),).join()

        # if user press "k", using autopilot mode
        elif vals[4] is True:
            # All is [[l_x, l_y], [r_x, r_y]]
            All = result[1]
            # DetectAreaAll is [[o_w, o_h]]
            DetectAreaAll = result[0]
            x = int((All[0, 0] + All[1, 0])/2)
            y = int((All[0, 1] + All[1, 1])/2)
            # fb is the speed drone move forward
            fb = 5
            # speedlr is the speed drone move left and right
            speedlr = 0
            # error is the distance of detecting box and the center of drone image
            errorX = x - w // 2
            errorY = y - h // 2

            # Calculate detect area
            AreaW = DetectAreaAll[0, 0]
            AreaH = DetectAreaAll[0, 1]
            Area = AreaW*AreaH

            if errorX > 0:
                # speedlr is the speed drone spin
                speedyv = 10
            else:
                speedyv = -10

            if errorY > -5:
                speedud = 3
            else:
                speedud = -3

            # if Detect area too small, drone fly straight. Protect drone hitting the wall.
            if Area < 15000:
                speedud = 0

            # airsim control command
            client.moveByVelocityBodyFrameAsync(fb, speedlr, speedud, 0.02,
                                                yaw_mode=airsim.YawMode(True, yaw_or_rate=speedyv), ).join()

def CameraAirSim(DetectResult):
    # Connect to drone image
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    cameraType = "scene"

    for arg in sys.argv[1:]:
      cameraType = arg.lower()

    cameraTypeMap = {
     "depth": airsim.ImageType.DepthVis,
     "segmentation": airsim.ImageType.Segmentation,
     "seg": airsim.ImageType.Segmentation,
     "scene": airsim.ImageType.Scene,
     "disparity": airsim.ImageType.DisparityNormalized,
     "normals": airsim.ImageType.SurfaceNormals
    }

    if (cameraType not in cameraTypeMap):
      printUsage()
      sys.exit(0)

    print (cameraTypeMap[cameraType])

    print("Connected: now while this script is running, you can open another")
    print("console and run a script that flies the drone and this script will")
    print("show the depth view while the drone is flying.")

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 2
    textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
    print(textSize)

    # Start YOLOv7-tiny add TensorRT
    device, binding_addrs, context, bindings, names, colors, win_title = TRTInitial()

    while True:
        # using in Detect Function
        ori_area = 0
        show_detect = False
        ##
        # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
        rawImage = client.simGetImage("0", cameraTypeMap[cameraType])
        if (rawImage == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            # Sometimes no image returns from api
            try:
                png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
            except:
                png = np.zeros((240, 360, 3), dtype=np.uint8)
            png, ori_area, All = Detect(png, device, binding_addrs, context, bindings, names, colors, win_title, ori_area, show_detect)
            result = [ori_area, All]
            DetectResult.put(result)

            cv2.imshow("YOLO", png)

        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q') or key == ord('x')):
            client.enableApiControl(False)
            break

# It is for calculate reward. Design an array to store sequential data across the target.
def Permutation(idx, sections):
    Permutation = np.empty((len(sections), 2), dtype=object)
    for j in range(len(sections)):
        section = sections[idx]
        idx += 1
        target_pos = section["target"]
        target_pos_yaw = section["offset5"][0]
        target = [target_pos]
        Permutation[j, 0] = target
        Permutation[j, 1] = target_pos_yaw
        if idx == len(sections):
            idx = 0

    return Permutation

# design for test your reward method.
def reward_calculate():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # Target distance based reward
    # Get train environment configs
    with open('config_test.yml', 'r') as f:
        env_config = yaml.safe_load(f)
    env_config = env_config["TrainEnv"]
    sections = env_config["sections"]
    target_idx = 0  # the first target number
    # Based on the first goal, list the remaining goals
    target_array = Permutation(target_idx, sections)
    idx = 0  # target array number

    # first target values
    target_first = target_array[idx][0]
    target_pos_yaw = target_array[idx][1]

    while True:
        # drone location
        x, y, z = client.simGetVehiclePose().position
        # distance of drone and target
        target_dist_curr = np.linalg.norm(np.array([x, y, z]) - target_first)

        # pass the target
        if target_dist_curr<5:
            # calculate final vector and calculate the last reward.
            # The closer the drone is to the center point, the higher score we get.
            x1 = math.cos(target_pos_yaw)
            y1 = math.sin(target_pos_yaw)
            z1 = 0

            target_normal_vector = np.array([x1, y1, z1])
            agent_target_vector = target_first - np.array([x, y, z])
            dot_product = np.dot(target_normal_vector, agent_target_vector[0])

            target_normal_vector_mag = np.linalg.norm(target_normal_vector)
            agent_target_vector_mag = np.linalg.norm(agent_target_vector)

            cos_angle = dot_product / (target_normal_vector_mag * agent_target_vector_mag)
            angle_rad = math.acos(np.clip(cos_angle, -1.0, 1.0))
            ###

            # reward value
            target_angle_point = (1 - abs(angle_rad))*10
            # print("target",target_angle_point)

            print("good ! : ", idx + 1)

            # changing target to the next one
            idx += 1
            target_first = target_array[idx][0]
            target_pos_yaw = target_array[idx][1]
            # if drone go through all target, idx need to return to zero.
            if idx == (len(sections)-1):
                idx = -1

        time.sleep(0.02)


## main program ##
if __name__ == '__main__':
    FlyAction = Queue(maxsize=1)
    DetectResult = Queue(maxsize=1)

    Thread1 = threading.Thread(target=KeyboardControl, args=(FlyAction,))
    Thread2 = threading.Thread(target=DroneFly, args=(FlyAction, DetectResult))
    Thread3 = threading.Thread(target=CameraAirSim, args=(DetectResult, ))
    Thread4 = threading.Thread(target=reward_calculate)

    Thread1.start()
    Thread2.start()
    Thread3.start()
    Thread4.start()

    Thread1.join()
    Thread2.join()
    Thread3.join()
    Thread4.join()


