import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

'''
0 Climbing-down
1 Climbing-up
2 Jumping
3 Lying
4 Running
5 Sitting  
6 Standing
7 Walking
'''

data = pd.read_csv(
    'C:/Users/karti/Downloads/HAR Files/30 Aug/dl-for-har-ea1e9babb2b178cc338dbc72db974325c193c781/dl-for-har'
    '-ea1e9babb2b178cc338dbc72db974325c193c781/results.csv')

pred = data[data.columns[0]].tolist()
gt = data[data.columns[1]].tolist()

pred = pred[:100000]
gt = gt[:100000]

myset = set(gt)
print(myset)

M = np.array([gt, pred])
plt.pcolor(M, cmap='hsv')
plt.xlabel("Window")
plt.ylabel("Ground Truth vs Predicted Values")
yellow_patch = mpatches.Patch(color='yellow', label='Climbing up')
pink_patch = mpatches.Patch(color='pink', label='Standing')
red_patch = mpatches.Patch(color='red', label='Walking')
dark_blue_patch = mpatches.Patch(color='purple', label='Sitting')
blue_patch = mpatches.Patch(color='blue', label='Running')
green_patch = mpatches.Patch(color='green', label='Climbing Down')
plt.legend(handles=[yellow_patch, pink_patch, red_patch, dark_blue_patch, blue_patch, green_patch], loc='upper left',
           bbox_to_anchor=(1, 1))
plt.show()

list_climb_up_idx_gt = []
list_climb_down_idx_gt = []
list_jump_idx_gt = []
list_lying_idx_gt = []
list_running_idx_gt = []
list_sitting_idx_gt = []
list_standing_idx_gt = []
list_walking_idx_gt = []
list_1 = []
list_2 = []
list_3 = []
list_4 = []
list_5 = []
list_6 = []
list_7 = []
list_8 = []

'''for i in range(len(gt)):
    if gt[i] == 1:
        list_climb_up_idx_gt.append(i)
        list_1.append(1)
    elif gt[i] == 2:
        list_climb_down_idx_gt.append(i)
        list_2.append(2)
    elif gt[i] == 3:
        list_jump_idx_gt.append(3)
        list_3.append(3)
    elif gt[i] == 4:
        list_lying_idx_gt.append(4)
        list_4.append(4)
    elif gt[i] == 5:
        list_running_idx_gt.append(5)
        list_5.append(5)
    elif gt[i] == 6:
        list_sitting_idx_gt.append(6)
        list_6.append(6)
    elif gt[i] == 7:
        list_standing_idx_gt.append(7)
        list_7.append(7)
    elif gt[i] == 8:
        list_walking_idx_gt.append(8)
        list_8.append(8)

print(len(list_2))

x = list_8
y = list_walking_idx_gt

plt.scatter(x, y, c='r')
plt.show()'''
