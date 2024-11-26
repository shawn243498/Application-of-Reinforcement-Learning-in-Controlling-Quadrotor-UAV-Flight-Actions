import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 定義曲線繪製函數
def plot_with_markers(x, y, T, number):
    # 繪製線條
    if number == 1:
        plt.plot(y, x, linestyle='--', linewidth=1, label='DQN_RIM')
    elif number == 2:
        plt.plot(y, x, linestyle='--', linewidth=1, label='DQN_RCM')
    elif number == 3:
        plt.plot(y, x, linewidth=2, label='DQN_CRM')
    elif number == 4:
        plt.plot(y, x, linestyle='--', linewidth=1, label='A2C_RIM')
    elif number == 5:
        plt.plot(y, x, linestyle='--', linewidth=1, label='A2C_RCM')
    elif number == 6:
        plt.plot(y, x, linestyle='--', linewidth=1, label='A2C_CRM')
    elif number == 7:
        plt.plot(y, x, linestyle='--', linewidth=1, label='PPO_RIM')
    elif number == 8:
        plt.plot(y, x, linestyle='--', linewidth=1, label='PPO_RCM')
    elif number == 9:
        plt.plot(y, x, linewidth=2, label='PPO_CRM')

    # 在起點加上紅色的點
    plt.scatter(y[0], x[0], color='red', zorder=5)

    for i in range(len(x)):
        if T[i] == 1:
            plt.plot(y[i], x[i], marker='o', color='green', markersize=8, zorder=5)

        #在終點加上叉叉
        if i == (len(x)-1):
            if T[i] == 0:
                plt.plot(y[i], x[i], marker='x', color='black', markersize=8, zorder=5)
            # elif T[i] == 1:
            #     plt.plot(x[i], y[i], marker='o', color='green', markersize=10, zorder=5)

df1 = pd.read_excel('../output_dqn_rim.xlsx')
# x1 = df1['X值']
y1 = df1['Y值']
z1 = df1['Z值']
T1 = df1['T']
number1 = 1

df2 = pd.read_excel('../output_dqn_rcm.xlsx')
# x2 = df2['X值']
y2 = df2['Y值']
z2 = df2['Z值']
T2 = df2['T']
number2 = 2

df3 = pd.read_excel('../output_dqn_crm.xlsx')
# x3 = df3['X值']
y3 = df3['Y值']
z3 = df3['Z值']
T3 = df3['T']
number3 = 3

df4 = pd.read_excel('../output_a2c_rim.xlsx')
# x4 = df4['X值']
y4 = df4['Y值']
z4 = df4['Z值']
T4 = df4['T']
number4 = 4

df5 = pd.read_excel('../output_a2c_rcm.xlsx')
# x5 = df5['X值']
y5 = df5['Y值']
z5 = df5['Z值']
T5 = df5['T']
number5 = 5

df6 = pd.read_excel('../output_a2c_crm.xlsx')
# x6 = df6['X值']
y6 = df6['Y值']
z6 = df6['Z值']
T6 = df6['T']
number6 = 6

df7 = pd.read_excel('../output_ppo_rim.xlsx')
# x7 = df7['X值']
y7 = df7['Y值']
z7 = df7['Z值']
T7 = df7['T']
number7 = 7

df8 = pd.read_excel('../output_ppo_rcm.xlsx')
# x8 = df8['X值']
y8 = df8['Y值']
z8 = df8['Z值']
T8 = df8['T']
number8 = 8

df9 = pd.read_excel('../output_ppo_crm.xlsx')
# df9 = pd.read_excel('output.xlsx')
# x9 = df9['X值']
y9 = df9['Y值']
z9 = df9['Z值']
T9 = df9['T']
number9 = 9
# plt.figure(figsize=(18, 5))

# 繪製所有線條
plot_with_markers(z1, y1, T1, number1)
plot_with_markers(z2, y2, T2, number2)
plot_with_markers(z3, y3, T3, number3)
plot_with_markers(z4, y4, T4, number4)
plot_with_markers(z5, y5, T5, number5)
plot_with_markers(z6, y6, T6, number6)
plot_with_markers(z7, y7, T7, number7)
plot_with_markers(z8, y8, T8, number8)
plot_with_markers(z9, y9, T9, number9)

# plt.title("Drone Trajectory Map", fontsize=18)
plt.xlabel("Y-axis", fontsize=16)
plt.ylabel("Z-axis", fontsize=16)

# 設置 X 軸和 Y 軸的刻度，每 20 單位一個刻度
plt.yticks(np.arange(-50, 0, 10), fontsize=14)
plt.xticks(np.arange(-160, 0, 20), fontsize=14)
# 反轉 X 軸和 Y 軸
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.gca().set_aspect('equal')

# plt.legend(fontsize=12)
plt.grid()
plt.show()