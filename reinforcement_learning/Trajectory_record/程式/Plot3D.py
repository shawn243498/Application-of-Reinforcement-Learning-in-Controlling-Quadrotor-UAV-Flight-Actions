import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

df1 = pd.read_excel('../output_dqn_rim.xlsx')
x1 = df1['X值']
y1 = df1['Y值']
z1 = df1['Z值']

df2 = pd.read_excel('../output_dqn_rcm.xlsx')
x2 = df2['X值']
y2 = df2['Y值']
z2 = df2['Z值']

df3 = pd.read_excel('../output_dqn_crm.xlsx')
x3 = df3['X值']
y3 = df3['Y值']
z3 = df3['Z值']

df4 = pd.read_excel('../output_a2c_rim.xlsx')
x4 = df4['X值']
y4 = df4['Y值']
z4 = df4['Z值']

df5 = pd.read_excel('../output_a2c_rcm.xlsx')
x5 = df5['X值']
y5 = df5['Y值']
z5 = df5['Z值']

df6 = pd.read_excel('../output_a2c_crm.xlsx')
x6 = df6['X值']
y6 = df6['Y值']
z6 = df6['Z值']

df7 = pd.read_excel('../output_ppo_rim.xlsx')
x7 = df7['X值']
y7 = df7['Y值']
z7 = df7['Z值']

df8 = pd.read_excel('../output_ppo_rcm.xlsx')
x8 = df8['X值']
y8 = df8['Y值']
z8 = df8['Z值']

df9 = pd.read_excel('../output_ppo_crm.xlsx')
x9 = df9['X值']
y9 = df9['Y值']
z9 = df9['Z值']

##################################

# 創建圖形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

##################################

# 繪製第一條軌跡
ax.plot(x1, y1, z1, linewidth=1, color='b', label='DQN_RIM')
ax.scatter(x1.iloc[0], y1.iloc[0], z1.iloc[0], color='r', s=100, marker='.')
ax.scatter(x1.iloc[-1], y1.iloc[-1], z1.iloc[-1], color='g', s=100, marker='.')

# 繪製第二條軌跡
ax.plot(x2, y2, z2, linewidth=1, color='orange', label='DQN_RCM')
ax.scatter(x2.iloc[0], y2.iloc[0], z2.iloc[0], color='r', s=100, marker='.')
ax.scatter(x2.iloc[-1], y2.iloc[-1], z2.iloc[-1], color='g', s=100, marker='.')

ax.plot(x3, y3, z3, linewidth=1, color='g', label='DQN_CRM')
ax.scatter(x3.iloc[0], y3.iloc[0], z3.iloc[0], color='r', s=100, marker='.')
ax.scatter(x3.iloc[-1], y3.iloc[-1], z3.iloc[-1], color='g', s=100, marker='.')

ax.plot(x4, y4, z4, linewidth=1, color='c', label='A2C_RIM')
ax.scatter(x4.iloc[0], y4.iloc[0], z4.iloc[0], color='r', s=100, marker='.')
ax.scatter(x4.iloc[-1], y4.iloc[-1], z4.iloc[-1], color='g', s=100, marker='.')

ax.plot(x5, y5, z5, linewidth=1, color='m', label='A2C_RCM')
ax.scatter(x5.iloc[0], y5.iloc[0], z5.iloc[0], color='r', s=100, marker='.')
ax.scatter(x5.iloc[-1], y5.iloc[-1], z5.iloc[-1], color='g', s=100, marker='.')

ax.plot(x6, y6, z6, linewidth=1, color='y', label='A2C_CRM')
ax.scatter(x6.iloc[0], y6.iloc[0], z6.iloc[0], color='r', s=100, marker='.')
ax.scatter(x6.iloc[-1], y6.iloc[-1], z6.iloc[-1], color='g', s=100, marker='.')

ax.plot(x7, y7, z7, linewidth=1, color='k', label='PPO_RIM')
ax.scatter(x7.iloc[0], y7.iloc[0], z7.iloc[0], color='r', s=100, marker='.')
ax.scatter(x7.iloc[-1], y7.iloc[-1], z7.iloc[-1], color='g', s=100, marker='.')

ax.plot(x8, y8, z8, linewidth=1, color='gold', label='PPO_RCM')
ax.scatter(x8.iloc[0], y8.iloc[0], z8.iloc[0], color='r', s=100, marker='.')
ax.scatter(x8.iloc[-1], y8.iloc[-1], z8.iloc[-1], color='g', s=100, marker='.')

ax.plot(x9, y9, z9, linewidth=1, color='lightcoral', label='PPO_CRM')
ax.scatter(x9.iloc[0], y9.iloc[0], z9.iloc[0], color='r', s=100, marker='.')
ax.scatter(x9.iloc[-1], y9.iloc[-1], z9.iloc[-1], color='g', s=100, marker='.')

####################################

# 設置標籤
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 反轉 Z 軸和 Y 軸
ax.invert_zaxis()
ax.invert_yaxis()

# 顯示圖例
ax.legend()

# 顯示圖形
plt.show()