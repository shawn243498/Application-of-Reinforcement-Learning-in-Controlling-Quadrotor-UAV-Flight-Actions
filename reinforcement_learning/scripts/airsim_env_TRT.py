from . import airsim
import gym
import numpy as np
import math
import time
### trt ###
import cv2
import torch
import random
import tensorrt as trt
from collections import OrderedDict,namedtuple

import pandas
class AirSimDroneEnv_TRT(gym.Env):
    def __init__(self, ip_address, image_shape, env_config):
        # ### trt param
        device, binding_addrs, context, bindings, names, colors, win_title = self.TRTInitial()
        self.device = device
        self.binding_addrs = binding_addrs
        self.context = context
        self.bindings = bindings
        self.names = names
        self.ori_area = 0
        self.show_detect = False
        ### trt param

        self.image_shape = image_shape
        print(image_shape)
        self.sections = env_config["sections"]   # Enter config file

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)  # Related to drone image input status
        self.action_space = gym.spaces.Discrete(9)  # Related to drone discrete action design

        self.info = {"collision": False}   # Drone collision parameters

        self.collision_time = 0
        self.random_start = False   # Determines whether the drone starts randomly.
                                    # It must be True during training and False during testing.
        self.count = 0     # The initial drone calculation parameters are intended to enable the drone to start its first training.
        self.out = False    # The drone passed through the target frame but was not recorded. When a certain score is greater than the set value,
                            # this value will be changed to True, causing the training loop to stop and the next training loop to start.
        self.target_pos_idx = -1   # Initialize the first target parameter

    def step(self, action):
        self.do_action(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        return obs, reward, done, info

    def reset(self):
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()

    # It is for calculate reward. Design an array to store sequential data across the target.
    def Permutation(self, idx):
        Permutation = np.empty((len(self.sections), 2), dtype=object)
        for j in range(len(self.sections)):
            section = self.sections[idx]
            idx += 1
            target_pos = section["target"]
            target_pos_yaw = section["offset5"][0]
            target = [target_pos, target_pos_yaw]
            Permutation[j] = target
            if idx == len(self.sections):
                idx = 0
        print(Permutation)
        print(len(self.sections))

        return Permutation

    def setup_flight(self):
        # The initialization of a single training loop of reinforcement learning operates
        # when the drone collides or when the initial parameters are zero.
        if self.is_collision() or self.count == 0:
            self.drone.reset()
            self.drone.enableApiControl(True)
            self.drone.armDisarm(True)

            # Prevent drone from falling after reset
            self.drone.moveToZAsync(-1, 1)

            # Get collision time stamp
            self.collision_time = self.drone.simGetCollisionInfo().time_stamp

            # Get a random section
            if self.random_start == True:   # For Training
                self.target_pos_idx = np.random.randint(len(self.sections))
            else:    # For Testing
                self.target_pos_idx += 1
                if self.target_pos_idx == 10:
                    self.target_pos_idx = 0


            # According to the first goal, arrange the remaining goals in order to obtain the goal matrix.
            self.target_array = self.Permutation(self.target_pos_idx)

            section = self.sections[self.target_pos_idx]
            self.agent_start_pos_x = section["offset1"][0]    # Target box center point : x
            self.agent_start_pos_y = section["offset2"][0]    # Target box center point : y
            self.agent_start_pos_z = section["offset3"][0]    # Target box center point : z
            self.agent_start_pos_yaw = section["offset4"][0]    # Target box center point : normal vector

            # Start the agent at random section at a random position
            pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos_x,self.agent_start_pos_y,self.agent_start_pos_z),
                               airsim.to_quaternion(0, 0, self.agent_start_pos_yaw))
            self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)

            self.idx = 0    # Order parameters in target matrix, When the drone crosses a target box, its value will be +1.
        
        # Get target distance for reward calculation
        self.out = False
        self.target_first = self.target_array[self.idx][0]      # Current target center point coordinates
        self.target_pos_yaw = self.target_array[self.idx][1]      # Current target center point normal vector
        # Distance of current target center point and drone.
        self.target_dist_prev = np.linalg.norm(np.array([self.agent_start_pos_x,self.agent_start_pos_y, self.agent_start_pos_z]) - self.target_first)

        self.step_num = 0   # calculate every episode how far the drone fly. It's for Reward.

        # Avoid self.count being 0, but you can change other writing methods.
        self.count += 1
        if self.count == 10:
            self.count = 1

    # Importance place!
    # Discrete motion design for drones.
    def do_action(self, select_action):
        speed1 = 3
        speed2 = 20
        # vz is the speed parameter that controls the rise and fall of the drone.
        # yaw is the speed parameter that controls the rotation of the drone.
        if select_action == 0:
            vz, yaw = (-speed1, -speed2)
        elif select_action == 1:
            vz, yaw = (0, -speed2)
        elif select_action == 2:
            vz, yaw = (speed1, -speed2)
        elif select_action == 3:
            vz, yaw = (-speed1, 0)
        elif select_action == 4:
            vz, yaw = (0, 0)
        elif select_action == 5:
            vz, yaw = (speed1, 0)
        elif select_action == 6:
            vz, yaw = (-speed1, speed2)
        elif select_action == 7:
            vz, yaw = (0, speed2)
        else:
            vz, yaw = (speed1, speed2)

        # Execute action
        self.drone.moveByVelocityBodyFrameAsync(5, 0, vz, duration=0.2,
                                                yaw_mode=airsim.YawMode(True, yaw_or_rate=yaw), ).join()

    def get_obs(self):
        self.info["collision"] = self.is_collision()
        obs = self.get_rgb_image()
        # cv2.imshow("123", obs)  # confirm obs is a image.
        return obs, self.info

    # Importance place!
    def compute_reward(self):
        reward = 0    # Initialize rewards for each timestep
        done = 0      # if done become 1 the episode will end.

        # Target distance based reward
        x,y,z = self.drone.simGetVehiclePose().position
        target_dist_curr = np.linalg.norm(np.array([x,y,z]) - self.target_first)
        if self.step_num != 0:
            reward += (self.target_dist_prev - target_dist_curr)*3     # self.target_dist_prev is the last distance of drone and target.
        self.target_dist_prev = target_dist_curr

        # Alignment reward
        # The further the drone flies, the more points will be deducted.
        self.step_num += 1
        reward += -self.step_num*0.05

        # When a drone collides.
        if self.is_collision():
            reward = -100
            done = 1
            print("collision !")
        # When a drone pass target.
        elif target_dist_curr < 5:
            print("target", self.target_pos_idx)

            # calculate final vector and calculate the last reward.
            # The closer the drone is to the center point, the higher score we get
            x1 = math.cos(self.target_pos_yaw)
            y1 = math.sin(self.target_pos_yaw)
            z1 = 0

            target_normal_vector = np.array([x1, y1, z1])
            agent_target_vector = self.target_first - np.array([x, y, z])
            dot_product = np.dot(target_normal_vector, agent_target_vector)

            target_normal_vector_mag = np.linalg.norm(target_normal_vector)
            agent_target_vector_mag = np.linalg.norm(agent_target_vector)

            cos_angle = dot_product / (target_normal_vector_mag * agent_target_vector_mag)
            angle_rad = math.acos(np.clip(cos_angle, -1.0, 1.0))

            target_angle_point = (1.57 - abs(angle_rad))*100/1.57
            print("target_angle_point", target_angle_point)
            reward = target_angle_point
            ##

            self.step_num = 0   # The number of drone flight steps for each Episode is reset to zero.

            self.idx += 1    # Update target number to next target.
            self.target_first = self.target_array[self.idx][0]
            self.target_pos_yaw = self.target_array[self.idx][1]
            if self.idx == (len(self.sections)-1):    # When the drone successfully flies over all targets, idx return to -1 and continue train.
                self.idx = -1

            done = 1
            print("good ! :")
        # Design of time compensation items.
        # When the drone flies over the target but is not recorded as crossing the target frame, this score will be used to skip this training loop.
        if reward < -3.5:
            done = 1
            print("no through hole")
            self.out = True
        print(reward)
        return reward, done

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time or self.out is True else False

################ trt ####################3
    def TRTInitial(self):
        print(trt.__version__)

        win_title = 'YOLOv7 tensorRT CUSTOM DETECTOR'

        w = './AirSimBest-nms.trt'
        device = torch.device('cuda:0')

        # Infer TensorRT Engine
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()

        names = ['Target', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                 'frisbee',
                 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard',
                 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone',
                 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                 'teddy bear',
                 'hair drier', 'toothbrush']
        colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}

        # warmup for 10 times
        for _ in range(5):
            tmp = torch.randn(1, 3, 640, 640).to(device)
            binding_addrs['images'] = int(tmp.data_ptr())
            context.execute_v2(list(binding_addrs.values()))

        return device, binding_addrs, context, bindings, names, colors, win_title

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def postprocess(self, boxes, r, dwdh):
        dwdh = torch.tensor(dwdh * 2).to(boxes.device)
        boxes -= dwdh
        boxes /= r
        return boxes

    def Detect(self, I, show_detect, ori_area):
        # t_prev = time.time()
        All = np.empty((2, 4), dtype=int)
        Center = np.empty((2, 2), dtype=int)
        OriArea2 = np.empty((2, 2), dtype=int)

        frame_rgb2 = np.zeros((240, 360, 3), dtype=np.uint8)
        try:
            frame_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        except:
            print("fix")
            frame_rgb = np.zeros((240, 360, 3), dtype=np.uint8)
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        image = frame_rgb.copy()  # is necessary but didn't know how it work
        image, ratio, dwdh = self.letterbox(image, auto=False)  # image preprocessing
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)

        im = torch.from_numpy(im).to(self.device)
        im /= 255

        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))

        nums = self.bindings['num_dets'].data
        boxes = self.bindings['det_boxes'].data
        scores = self.bindings['det_scores'].data
        classes = self.bindings['det_classes'].data

        boxes = boxes[0, :nums[0][0]]
        scores = scores[0, :nums[0][0]]
        classes = classes[0, :nums[0][0]]

        all_targets = []

        for box, score, cl in zip(boxes, scores, classes):
            box = self.postprocess(box, ratio, dwdh).round().int()
            name = self.names[cl]

            if name == 'Target' and score > 0.6:
                show_detect = True
                name += ' ' + str(round(float(score), 3))

                l_x = int(box[0])
                l_y = int(box[1])
                r_x = int(box[2])
                r_y = int(box[3])
                o_w = int(r_x - l_x)
                o_h = int(r_y - l_y)
                area = o_w * o_h

        #### MTTM ####
                all_targets.append({
                    'box': [l_x, l_y, r_x, r_y],
                    'center': [(l_x + r_x) // 2, (l_y + r_y) // 2],
                    'area': area,
                    'score': score
                })

        # 根據面積對目標進行排序，選擇最大的兩個目標
        all_targets.sort(key=lambda x: x['area'])
        top_two_targets = all_targets[-2:]

        # 繪製最大的兩個目標
        for i, target in enumerate(top_two_targets):
            l_x, l_y, r_x, r_y = target['box']
            center_x, center_y = target['center']
            area = target['area']
            score = target['score']

            OriArea2[i, 0] = r_x - l_x
            OriArea2[i, 1] = r_y - l_y

            All[i, 0] = l_x
            All[i, 1] = l_y
            All[i, 2] = r_x
            All[i, 3] = r_y

            if len(top_two_targets) == 1:
                if i == 0:
                    cv2.circle(frame_rgb2, (center_x, center_y), 5, (0, 255, 255), -1)
                    cv2.rectangle(frame_rgb2, (l_x, l_y), (r_x, r_y), (0, 255, 255), 2)
            else:
                if i == 0:
                    cv2.circle(frame_rgb2, (center_x, center_y), 5, (255, 0, 255), -1)
                    cv2.rectangle(frame_rgb2, (l_x, l_y), (r_x, r_y), (255, 0, 255), 2)
                if i == 1:
                    cv2.circle(frame_rgb2, (center_x, center_y), 5, (0, 255, 255), -1)
                    cv2.rectangle(frame_rgb2, (l_x, l_y), (r_x, r_y), (0, 255, 255), 2)

        # ####### MSTM ############
        #         if area > ori_area:
        #             ori_area = area
        #             OriArea2[0, 0] = o_w
        #             OriArea2[0, 1] = o_h
        #             All[0, 0] = l_x
        #             All[0, 1] = l_y
        #             All[1, 0] = r_x
        #             All[1, 1] = r_y
        #             CenterX = int((l_x + r_x) / 2)
        #             CenterY = int((l_y + r_y) / 2)
        #             Center[0, 0] = CenterX
        #             Center[0, 1] = CenterY
        #
        # if show_detect is True:
        #     cv2.circle(frame_rgb2, (Center[0, 0], Center[0, 1]), 5, (255, 0, 255), -1)
        #     cv2.rectangle(frame_rgb2, (All[0, 0], All[0, 1]), (All[1, 0], All[1, 1]), (0, 255, 255), 2)
        ################
        img2 = cv2.cvtColor(frame_rgb2, cv2.COLOR_BGR2RGB)

        return img2

################ trt ####################3
    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))

        ############### add trt ############3
        # cv2.imshow("123", img2d) # 確定img2d是一張圖片
        show_detect = self.show_detect
        ori_area = self.ori_area
        img2d = self.Detect(img2d, show_detect, ori_area)
        #
        # cv2.imshow("123", img2d)  # 確定img2d是一張圖片
        ############### add trt ############3

        # Sometimes no image returns from api
        try:
            img2d = img2d.reshape(self.image_shape)
            cv2.imshow("123", img2d)
            cv2.waitKey(1)
            return img2d
        except:
            return np.zeros((self.image_shape))

    # Not used, if you want to use depth you can try it
    def get_depth_image(self, thresh = 2.0):
        depth_image_request = airsim.ImageRequest(1, airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image>thresh]=thresh
        return depth_image


class TestEnv_TRT(AirSimDroneEnv_TRT):
    def __init__(self, ip_address, image_shape, env_config):
        self.eps_n = 0    # number of testing round
        # __init__ in the AirSimDroneEnv_TRT function will be executed once
        super(TestEnv_TRT, self).__init__(ip_address, image_shape, env_config)

        # Record drone position in flight
        self.numbersX = []
        self.numbersY = []
        self.numbersZ = []
        # Record the point drone pass the target
        self.Touch = []
        # Record the number drone pass the target
        self.count_pass = 0

        ## last
        self.mean_total_target_go_through_array = []
        self.total_target_collision = 0
        self.mean_flight_time_array = []
        self.whole_field = 0
        self.go_out = 0
        self.num_col = []
        self.num_out = []
        self.num_pass = []

        ## every episode
        self.eps_n = 0
        self.target_go_through = 0
        self.flight_time = 0
        self.flight_time_array = []
        self.episode_flight_time = 0
        self.t1 = 0
        self.t2 = 0

    def setup_flight(self):
        self.t1 = time.time()
        super(TestEnv_TRT, self).setup_flight()
        
    def compute_reward(self):
        done = 0
        end = 0

        x,y,z = self.drone.simGetVehiclePose().position

        x = round(x, 3)
        y = round(y, 3)
        z = round(z, 3)
        t = 0

        self.numbersX.append(x)
        self.numbersY.append(y)
        self.numbersZ.append(z)

        self.step_num += 1
        target_dist_curr = np.linalg.norm(np.array([x, y, z]) - self.target_first)

        if self.is_collision():
            self.t2 = time.time()
            self.flight_time = self.t2 - self.t1
            if len(self.flight_time_array) > 0:
                self.episode_flight_time = sum(self.flight_time_array) / len(self.flight_time_array)
            self.mean_total_target_go_through_array.append(self.target_go_through)
            self.num_col.append(self.eps_n)

            done = 1
            self.step_num = 0
            self.eps_n += 1
            self.total_target_collision += 1
            print("bad !")

            ## test
            print("self.flight_time_array : ", self.flight_time_array)
            print("self.flight_time_array : ", len(self.flight_time_array))

            ## every episode
            print("---------------------------------")
            print("> Episodes:", self.eps_n)  ### 1~11
            print("> Holes reached :", self.target_go_through)  ### 1~11
            print("> Mean Flight time : %.2f" % self.episode_flight_time)
            print("> Collision ? : Yes")
            print("---------------------------------\n")

            ## Return to zero
            self.count_pass = 0
            self.target_go_through = 0
            self.flight_time_array = []
            self.episode_flight_time = 0

            ## final
            if self.eps_n == 11:
                ## test
                print("self.mean_flight_time_array : ", self.mean_flight_time_array)
                print("self.mean_flight_time_array : ", len(self.mean_flight_time_array))
                print("self.mean_total_target_go_through_array", self.mean_total_target_go_through_array)
                print("self.mean_total_target_go_through_array", sum(self.mean_total_target_go_through_array))

                if len(self.mean_flight_time_array) > 0:
                    total_mean_flight_time = sum(self.mean_flight_time_array) / len(self.mean_flight_time_array)
                else:
                    total_mean_flight_time = 0

                if len(self.mean_total_target_go_through_array) > 0:
                    mean_total_target_go_through = sum(self.mean_total_target_go_through_array) / len(
                        self.mean_total_target_go_through_array)
                else:
                    mean_total_target_go_through = 0

                print("---------------------------------")
                print("> Holes reached (max, mean):", sum(self.mean_total_target_go_through_array),
                      mean_total_target_go_through)  ### 1~11
                print("> Flight time (mean): %.2f" % total_mean_flight_time)
                print("> Collision time :", self.total_target_collision, self.num_col)
                print("> Go Out time :", self.go_out, self.num_out)
                print("> Pass Field time  :", self.whole_field, self.num_pass)
                print("---------------------------------\n")

                self.Touch.append(t)
                df = pandas.DataFrame({'X值': self.numbersX, 'Y值': self.numbersY, 'Z值': self.numbersZ, 'T':self.Touch})
                df.to_excel('output.xlsx', index=False, engine='openpyxl')
                print("Value has been written successfully in 'output.xlsx'")
                end = 1


        elif target_dist_curr < 8:
            self.t2 = time.time()
            self.flight_time = self.t2 - self.t1
            self.flight_time_array.append(self.flight_time)
            self.target_go_through += 1

            t = 1

            done = 1
            self.step_num = 0
            self.idx += 1
            self.target_first = self.target_array[self.idx][0]
            self.target_pos_yaw = self.target_array[self.idx][1]
            if self.idx == (len(self.sections) - 1):
                self.idx = -1
                # done = 1
            print("good !", self.target_go_through)
            self.count_pass += 1

            print("> Flight time : %.2f" % self.flight_time)
            if len(self.flight_time_array) > 0:
                self.episode_flight_time = sum(self.flight_time_array) / len(self.flight_time_array)
                self.mean_flight_time_array.append(self.flight_time)

            if self.count_pass == 11:
                self.mean_total_target_go_through_array.append(self.target_go_through)
                self.num_pass.append(self.eps_n)
                self.whole_field += 1
                self.out = True
                self.eps_n += 1

                ## test
                print("self.flight_time_array : ", self.flight_time_array)
                print("self.flight_time_array : ", len(self.flight_time_array))

                ## every episode
                print("---------------------------------")
                print("> Episodes:", self.eps_n)  ### 1~11
                print("> Holes reached :", self.target_go_through)  ### 1~11
                print("> Mean Flight time : %.2f" % self.episode_flight_time)
                print("> Collision ? : No, Success Whole Field")
                print("---------------------------------\n")

                ## Return to zero
                self.count_pass = 0
                self.target_go_through = 0
                self.flight_time_array = []
                self.episode_flight_time = 0

                ## final
                if self.eps_n == 11:
                    ## test
                    print("self.mean_flight_time_array : ", self.mean_flight_time_array)
                    print("self.mean_flight_time_array : ", len(self.mean_flight_time_array))
                    print("self.mean_total_target_go_through_array", self.mean_total_target_go_through_array)
                    print("self.mean_total_target_go_through_array", sum(self.mean_total_target_go_through_array))

                    if len(self.mean_flight_time_array) > 0:
                        total_mean_flight_time = sum(self.mean_flight_time_array) / len(self.mean_flight_time_array)
                    else:
                        total_mean_flight_time = 0

                    if len(self.mean_total_target_go_through_array) > 0:
                        mean_total_target_go_through = sum(self.mean_total_target_go_through_array) / len(
                            self.mean_total_target_go_through_array)
                    else:
                        mean_total_target_go_through = 0

                    print("---------------------------------")
                    print("> Holes reached (max, mean):", sum(self.mean_total_target_go_through_array),
                          mean_total_target_go_through)  ### 1~11
                    print("> Flight time (mean): %.2f" % total_mean_flight_time)
                    print("> Collision time :", self.total_target_collision, self.num_col)
                    print("> Go Out time :", self.go_out, self.num_out)
                    print("> Pass Field time  :", self.whole_field, self.num_pass)
                    print("---------------------------------\n")

                    self.Touch.append(t)
                    df = pandas.DataFrame({'X值': self.numbersX, 'Y值': self.numbersY, 'Z值': self.numbersZ, 'T':self.Touch})
                    df.to_excel('output.xlsx', index=False, engine='openpyxl')
                    print("Value has been written successfully in 'output.xlsx'")
                    end = 1

            # elif reward < -3.5:
        elif self.step_num > 40:
            self.t2 = time.time()
            self.flight_time = self.t2 - self.t1
            if len(self.flight_time_array) > 0:
                self.episode_flight_time = sum(self.flight_time_array) / len(self.flight_time_array)
            self.mean_total_target_go_through_array.append(self.target_go_through)
            self.num_out.append(self.eps_n)
            self.go_out += 1

            done = 1
            self.step_num = 0
            self.out = True
            self.eps_n += 1
            print("no through hole !")

            ## test
            print("self.flight_time_array : ", self.flight_time_array)
            print("self.flight_time_array : ", len(self.flight_time_array))

            ## every episode
            print("---------------------------------")
            print("> Episodes:", self.eps_n)  ### 1~11
            print("> Holes reached :", self.target_go_through)  ### 1~11
            print("> Mean Flight time : %.2f" % self.episode_flight_time)
            print("> Collision ? : No But Out")
            print("---------------------------------\n")

            ## Return to zero
            self.count_pass = 0
            self.target_go_through = 0
            self.flight_time_array = []
            self.episode_flight_time = 0

            ## final
            if self.eps_n == 11:
                ## test
                print("self.mean_flight_time_array : ", self.mean_flight_time_array)
                print("self.mean_flight_time_array : ", len(self.mean_flight_time_array))
                print("self.mean_total_target_go_through_array", self.mean_total_target_go_through_array)
                print("self.mean_total_target_go_through_array", sum(self.mean_total_target_go_through_array))

                if len(self.mean_flight_time_array) > 0:
                    total_mean_flight_time = sum(self.mean_flight_time_array) / len(self.mean_flight_time_array)
                else:
                    total_mean_flight_time = 0

                if len(self.mean_total_target_go_through_array) > 0:
                    mean_total_target_go_through = sum(self.mean_total_target_go_through_array) / len(
                        self.mean_total_target_go_through_array)
                else:
                    mean_total_target_go_through = 0

                print("---------------------------------")
                print("> Holes reached (max, mean):", sum(self.mean_total_target_go_through_array),
                      mean_total_target_go_through)  ### 1~11
                print("> Flight time (mean): %.2f" % total_mean_flight_time)
                print("> Collision time :", self.total_target_collision, self.num_col)
                print("> Go Out time :", self.go_out, self.num_out)
                print("> Pass Field time  :", self.whole_field, self.num_pass)
                print("---------------------------------\n")

                self.Touch.append(t)
                df = pandas.DataFrame({'X值': self.numbersX, 'Y值': self.numbersY, 'Z值':self.numbersZ, 'T':self.Touch})
                df.to_excel('output.xlsx', index=False, engine='openpyxl')
                print("Value has been written successfully in 'output.xlsx'")
                end = 1

        self.Touch.append(t)
        return end, done
