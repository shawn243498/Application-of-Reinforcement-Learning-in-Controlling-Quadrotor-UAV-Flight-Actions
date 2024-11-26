import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

pose = airsim.Pose(airsim.Vector3r(-0.15, -27.5, -11),
                   airsim.to_quaternion(0, 0, -1.05))
client.simSetVehiclePose(pose=pose, ignore_collision=True)

client.moveToZAsync(-1, 1)
client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

#################################################################

# import numpy as np
#
# vectorX1 = -0.15
# vectorY1 = -27.5
# vectorZ1 = -11
#
# vectorX2 = -41.2
# vectorY2 = -7.9
# vectorZ2 = -16
#
# # 一个三维向量
# vector = np.array([vectorX2-vectorX1, vectorY2-vectorY1, vectorZ2-vectorZ1])
#
# # 计算向量的长度
# length = np.linalg.norm(vector)
#
# print("向量的长度（欧几里得距离）为:", length)

##################################################################
# import numpy as np
#
#
# def angle_between_lines(P1, P2, P3):
#     # 将点转换为numpy数组
#     P1 = np.array(P1)
#     P2 = np.array(P2)
#     P3 = np.array(P3)
#
#     # 计算P1到P2和P1到P3的向量
#     vector1 = P2 - P1
#     vector2 = P3 - P1
#
#     # 计算向量的点积和模长
#     dot_product = np.dot(vector1, vector2)
#     norm_vector1 = np.linalg.norm(vector1)
#     norm_vector2 = np.linalg.norm(vector2)
#
#     # 计算夹角的余弦值
#     cos_angle = dot_product / (norm_vector1 * norm_vector2)
#
#     # 计算夹角（弧度）
#     angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))
#
#     # 将弧度转换为角度
#     angle_degrees = np.degrees(angle_radians)
#
#     return angle_degrees
#
#
# # 三个点的示例
# P1 = [1, 2, 3]
# P2 = [2, 3, 4]
# P3 = [7, 4, 5]
#
# # 计算夹角
# angle = angle_between_lines(P1, P2, P3)
# print("P2和P1形成的直线和P3和P1形成的直线的夹角为：", angle, "度")

#
# import math
#
# def rotate_z_axis(theta):
#     # 将角度转换为弧度
#     theta_rad = theta
#
#     # 计算旋转后的法向量分量
#     x = math.cos(theta_rad)
#     y = math.sin(theta_rad)
#     z = 0
#
#     # 返回单位法向量
#     return [x, y, z]
#
#
# # 输入旋转角度
# theta = float(input("请输入旋转角度（度）："))
#
# # 计算并打印法向量
# normal_vector = rotate_z_axis(theta)
# print("旋转后的单位法向量为：", normal_vector)


###### 之前用過的程式
# ######################################### calculate angle
# # print("agent_traveled_All", agent_traveled_All)
# if agent_traveled_All > 5:
#     P1 = np.array([self.agent_start_pos_x,
#                 self.agent_start_pos_y,
#                 self.agent_start_pos_z])
#     P2 = self.target_pos
#     P3 = np.array([x,y,z])
#
#     vector1 = P2 - P1
#     vector2 = P3 - P1
#     # 计算向量的点积和模长
#     dot_product = np.dot(vector1, vector2)
#     norm_vector1 = np.linalg.norm(vector1)
#     norm_vector2 = np.linalg.norm(vector2)
#
#     # 计算夹角的余弦值
#     cos_angle = dot_product / (norm_vector1 * norm_vector2)
#
#     # 计算夹角（弧度）
#     angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))
#
# ######################################## calculate angle
#
# ######################################## calculate plane
# theta_rad = self.target_pos_yaw
#
# # 计算旋转后的法向量分量
# x1 = math.cos(theta_rad)
# y1 = math.sin(theta_rad)
# z1 = 0
#
# x2 = self.target_pos[0] + 3.88*math.cos(theta_rad)
# y2 = self.target_pos[1] + 3.88*math.sin(theta_rad)
# z2 = self.target_pos[2]
#
# target_normal_vector = np.array([x1, y1, z1])
# agent_target_vector = np.array([x,y,z])-np.array([x2, y2, z2])
#
# # 計算兩向量的內積
# dot_product2 = np.dot(target_normal_vector, agent_target_vector)
#
# # 計算兩向量的模
# target_normal_vector_mag = np.linalg.norm(target_normal_vector)
# agent_target_vector_mag = np.linalg.norm(agent_target_vector)
#
# # 計算夾角（以弧度為單位）
# cos_angle2 = dot_product2 / (target_normal_vector_mag * agent_target_vector_mag)
# angle_rad2 = math.acos(np.clip(cos_angle2, -1.0, 1.0))
# # print("angle_rad2", angle_rad2)
# ######################################## calculate plane

# #### if angle > 30 degree
# if agent_traveled_All > 15:
#     # print("angle_radians", angle_radians)
#     reward += -0.2
#     if angle_radians > 0.785:
#         reward += -100
#         done = 1
#         print("bad1 ! :", reward)
#
# if angle_rad2 < 1.57:
#     reward += -100
#     done = 1
#     print("bad2 ! :", reward)


