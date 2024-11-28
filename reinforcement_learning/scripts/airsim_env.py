from . import airsim
import gym
import numpy as np
import math
import time

import cv2

import pandas
class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape, env_config):
        self.image_shape = image_shape
        print(image_shape)
        self.sections = env_config["sections"]

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(9)

        self.info = {"collision": False}

        self.collision_time = 0
        self.random_start = True
        self.count = 0
        self.out = False
        self.target_pos_idx = -1

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
        if self.is_collision() or self.count == 0:
            self.drone.reset()
            self.drone.enableApiControl(True)
            self.drone.armDisarm(True)

            # Prevent drone from falling after reset
            self.drone.moveToZAsync(-1, 1)

            # Get collision time stamp
            self.collision_time = self.drone.simGetCollisionInfo().time_stamp

            # Get a random section
            if self.random_start == True:
                self.target_pos_idx = np.random.randint(len(self.sections))
            else:
                self.target_pos_idx += 1
                if self.target_pos_idx == 10:
                    self.target_pos_idx = 0

            ##### number the target #####
            self.target_array = self.Permutation(self.target_pos_idx)
            ##### number the target #####

            section = self.sections[self.target_pos_idx]
            self.agent_start_pos_x = section["offset1"][0]
            self.agent_start_pos_y = section["offset2"][0]
            self.agent_start_pos_z = section["offset3"][0]
            self.agent_start_pos_yaw = section["offset4"][0]

            # Start the agent at random section at a random yz position
            pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos_x,self.agent_start_pos_y,self.agent_start_pos_z),
                               airsim.to_quaternion(0, 0, self.agent_start_pos_yaw))
            self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)

            self.idx = 0
        
        # Get target distance for reward calculation
        self.out = False
        self.target_first = self.target_array[self.idx][0]
        self.target_pos_yaw = self.target_array[self.idx][1]
        self.target_dist_prev = np.linalg.norm(np.array([self.agent_start_pos_x,self.agent_start_pos_y, self.agent_start_pos_z]) - self.target_first)

        self.step_num = 0

        self.count += 1
        if self.count == 10:
            self.count = 1

    def do_action(self, select_action):
        speed1 = 3
        speed2 = 20
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

    def compute_reward(self):
        reward = 0
        done = 0

        # Target distance based reward
        x,y,z = self.drone.simGetVehiclePose().position
        target_dist_curr = np.linalg.norm(np.array([x,y,z]) - self.target_first)
        if self.step_num != 0:
            reward += (self.target_dist_prev - target_dist_curr)*3

        self.target_dist_prev = target_dist_curr

        # Alignment reward
        ## step point
        self.step_num += 1
        reward += -self.step_num*0.05

        if self.is_collision():
            reward = -100
            done = 1
            print("collision !")

        elif target_dist_curr < 5:
            print("target", self.target_pos_idx)

            ##### cal end vector
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
            ##### cal end vector

            target_angle_point = (1.57 - abs(angle_rad))*100/1.57
            print("target_angle_point", target_angle_point)
            reward = target_angle_point

            self.step_num = 0

            self.idx += 1
            self.target_first = self.target_array[self.idx][0]
            self.target_pos_yaw = self.target_array[self.idx][1]

            done = 1
            # reward = 100

            if self.idx == (len(self.sections)-1):
                self.idx = -1
            print("good ! :")

        if reward < -3.5:
            done = 1
            print("no through hole")
            self.out = True
        print(reward)
        return reward, done

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time or self.out is True else False
    
    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))

        # Sometimes no image returns from api
        try:
            img2d = img2d.reshape(self.image_shape)
            cv2.imshow("123", img2d)
            cv2.waitKey(1)
            return img2d
        except:
            return np.zeros((self.image_shape))

    def get_depth_image(self, thresh = 2.0):
        depth_image_request = airsim.ImageRequest(1, airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = np.reshape(depth_image, (responses[0].height, responses[0].width))
        depth_image[depth_image>thresh]=thresh
        return depth_image


class TestEnv(AirSimDroneEnv):
    def __init__(self, ip_address, image_shape, env_config):
        self.eps_n = 0
        super(TestEnv, self).__init__(ip_address, image_shape, env_config)
        self.random_start = False
        self.numbersX = []
        self.numbersY = []
        self.numbersZ = []
        self.Touch = []

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
        super(TestEnv, self).setup_flight()
        # self.eps_n += 1
        
    def compute_reward(self):
        done = 0
        end = 0

        x,y,z = self.drone.simGetVehiclePose().position

        x = round(x, 3)
        y = round(y, 3)
        z = round(z, 3)
        t = 0
        # print(x,y,z)

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
                print("數值已成功寫入 'output.xlsx'")
                end = 1


        elif target_dist_curr < 8:
            print(x, y, z)

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
                    print("數值已成功寫入 'output.xlsx'")
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
                print("數值已成功寫入 'output.xlsx'")
                end = 1

        self.Touch.append(t)
        return end, done
