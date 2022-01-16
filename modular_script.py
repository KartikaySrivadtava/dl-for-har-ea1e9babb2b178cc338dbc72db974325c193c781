import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FACTOR = 4


def load_data():
    df = pd.read_csv('C:/Users/karti/PycharmProjects/HAR/data/rwhar_data.csv', sep=',')
    print(df.head())
    return df


def resampling(df, n):
    print("Inside resampling")
    print(df.head())
    print("resampling factor is ", n)

    # Down-sampling
    df_down = downsampling(df, n)

    # Up-sampling
    df_up = upsampling(df, df_down, n)

    # Down-Sampling
    df_down = downsampling(df_up, n)

    # Up-Sampling
    df_up = upsampling(df, df_down, n)

    # Down-Sampling
    df_down = downsampling(df_up, n)

    # Up-Sampling
    df_up = upsampling(df, df_down, n)

    # Down-Sampling
    df_down = downsampling(df_up, n)

    # Up-Sampling
    df_up = upsampling(df, df_down, n)


def downsampling(df, n):
    df_down = df[df.index % n == 0]
    df_down.to_csv('C:/Users/karti/PycharmProjects/HAR/data/rwhar_data_down.csv', header=False, index=False)
    print("downsampled file created successfully")
    return df_down


def upsampling(df, df_down, n):
    list_1 = df['sub'].to_numpy()
    print("shape for sub is ", list_1.shape)
    list_2 = df_down['x'].to_numpy()
    print("shape for x is ", list_2.shape)
    list_3 = df_down['y'].to_numpy()
    print("shape for y is ", list_3.shape)
    list_4 = df_down['z'].to_numpy()
    print("shape for z is ", list_4.shape)
    list_5 = df['act'].to_numpy()
    print("shape for act is ", list_5.shape)

    list_2_new = []

    for i in range(list_2.size - 1):
        list_2_new.append((np.linspace(list_2[i], list_2[i + 1], n)))

    np_list_2_new = (np.array(list_2_new)).flatten()
    # print(np_list_2_new)

    list_3_new = []

    for i in range(list_3.size - 1):
        list_3_new.append((np.linspace(list_3[i], list_3[i + 1], n)))

    np_list_3_new = (np.array(list_3_new)).flatten()
    # print(np_list_3_new)

    list_4_new = []

    for i in range(list_4.size - 1):
        list_4_new.append((np.linspace(list_4[i], list_4[i + 1], n)))

    np_list_4_new = (np.array(list_4_new)).flatten()
    # print(np_list_4_new)
    np.resize(list_1, list_1.size - 3)
    np.resize(list_5, list_5.size - 3)
    print("new size for x is ", np_list_2_new.size)
    print("new size for y is ", np_list_3_new.size)
    print("new size for z is ", np_list_4_new.size)
    print("size for sub is ", list_1.size)
    print("size for act is ", list_5.size)

    df_up = pd.DataFrame(list(zip(list_1, np_list_2_new, np_list_3_new, np_list_4_new, list_5)),
                         columns=['sub', 'x', 'y', 'z', 'act'])
    df_up.to_csv('C:/Users/karti/PycharmProjects/HAR/data/rwhar_data_up.csv', header=False, index=False)
    print("up-sampled dataframe created successfully")
    return df_up


def evaluation():
    print("Inside evaluation")
    '''eval_df = pd.read_csv('C:/Users/karti/PycharmProjects/HAR/data/downsampled_data/f1_final.csv', sep=',',
                          index_col=None, header=None)'''
    # print(eval_df)
    '''df_climb_down = eval_df.loc[eval_df[1] == 'climbingdown']
    df_climb_up = eval_df.loc[eval_df[1] == 'climbingup']
    df_jumping = eval_df.loc[eval_df[1] == 'jumping']
    df_lying = eval_df.loc[eval_df[1] == 'lying']
    df_running = eval_df.loc[eval_df[1] == 'running']
    df_sitting = eval_df.loc[eval_df[1] == 'sitting']
    df_standing = eval_df.loc[eval_df[1] == 'standing']
    df_waling = eval_df.loc[eval_df[1] == 'walking']
    list_act = ['50Hz', '25Hz', '12.5', '6.25Hz', '3.125Hz']
    list_climb_down = df_climb_down[0]
    list_climb_up = df_climb_up[0]
    list_jumping = df_jumping[0]
    list_lying = df_lying[0]
    list_running = df_running[0]
    list_sitting = df_sitting[0]
    list_standing = df_standing[0]
    list_walking = df_waling[0]
    # print(list_climb_up)
    dict_climb_up = dict(zip(list_act, list_climb_up))
    dict_climb_down = dict(zip(list_act, list_climb_down))
    dict_jump = dict(zip(list_act, list_jumping))
    dict_lying = dict(zip(list_act, list_lying))
    dict_run = dict(zip(list_act, list_running))
    dict_sit = dict(zip(list_act, list_sitting))
    dict_stand = dict(zip(list_act, list_standing))
    dict_walk = dict(zip(list_act, list_walking))

    dict_climb_up = dict(reversed(list(dict_climb_up.items())))
    dict_climb_down = dict(reversed(list(dict_climb_down.items())))
    dict_jump = dict(reversed(list(dict_jump.items())))
    dict_lying = dict(reversed(list(dict_lying.items())))
    dict_run = dict(reversed(list(dict_run.items())))
    dict_sit = dict(reversed(list(dict_sit.items())))
    dict_stand = dict(reversed(list(dict_stand.items())))
    dict_walk = dict(reversed(list(dict_walk.items())))

    print("Dictionary for climb up", dict_climb_up)
    print("Dictionary for climb down", dict_climb_down)
    print("Dictionary for jumping", dict_jump)
    print("Dictionary for lying", dict_lying)
    print("Dictionary for running", dict_run)
    print("Dictionary for sitting", dict_sit)
    print("Dictionary for standing", dict_stand)
    print("Dictionary for walking", dict_walk)

    # Climbing up
    x = np.array(["3.125Hz", "6.25Hz", "12.5Hz", "25Hz", "50Hz"])
    y = np.array([0.16120577, 1.46228751e-01, 0.12200513, 0.16865588, 0.17731791])
    ax1 = plt.subplot(2, 4, 1)
    ax1.set_ylim([0, 1])
    ax1.bar(x, y)
    ax1.set_title('F1 climbing up')

    # Climbing down
    x = np.array(["3.125Hz", "6.25Hz", "12.5Hz", "25Hz", "50Hz"])
    y = np.array([0.00894569, 6.42054575e-04, 0.00737898, 0.55602312, 0.50007438])
    ax2 = plt.subplot(2, 4, 2)
    ax2.bar(x, y)
    ax2.set_yticklabels([])
    ax2.set_ylim([0, 1])
    ax2.set_title('F1 climbing down')

    # Jumping
    x = np.array(["3.125Hz", "6.25Hz", "12.5Hz", "25Hz", "50Hz"])
    y = np.array([0.0, 2.80112045e-03, 0.07974684, 0.79935012, 0.93278539])
    ax3 = plt.subplot(2, 4, 3)
    ax3.bar(x, y)
    ax3.set_yticklabels([])
    ax3.set_ylim([0, 1])
    ax3.set_title('F1 jumping')

    # Lying
    x = np.array(["3.125Hz", "6.25Hz", "12.5Hz", "25Hz", "50Hz"])
    y = np.array([0.69628775, 6.86327992e-01, 0.71235705, 0.72834384, 0.71312762])

    ax4 = plt.subplot(2, 4, 4)
    ax4.bar(x, y)
    ax4.set_yticklabels([])
    ax4.set_ylim([0, 1])
    ax4.set_title('F1 lying')

    # Running
    x = np.array(["3.125Hz", "6.25Hz", "12.5Hz", "25Hz", "50Hz"])
    y = np.array([0.74904858, 4.12183055e-01, 0.68574628, 0.87960257, 0.90076908])

    ax5 = plt.subplot(2, 4, 5)
    ax5.bar(x, y)
    ax5.set_yticklabels([])
    ax5.set_ylim([0, 1])
    ax5.set_title('F1 running')

    # Sitting
    x = np.array(["3.125Hz", "6.25Hz", "12.5Hz", "25Hz", "50Hz"])
    y = np.array([0.57946871, 5.39011634e-01, 0.54231007, 0.53954559, 0.5122115])

    ax6 = plt.subplot(2, 4, 6)
    ax6.bar(x, y)
    ax6.set_yticklabels([])
    ax6.set_ylim([0, 1])
    ax6.set_title('F1 Sitting')

    # Standing
    x = np.array(["3.125Hz", "6.25Hz", "12.5Hz", "25Hz", "50Hz"])
    y = np.array([0.45651606, 4.15405139e-01, 0.66609605, 0.77698193, 0.76365756])

    ax7 = plt.subplot(2, 4, 7)
    ax7.bar(x, y)
    ax7.set_yticklabels([])
    ax7.set_ylim([0, 1])
    ax7.set_title('F1 Standing')

    # Walking
    x = np.array(["3.125Hz", "6.25Hz", "12.5Hz", "25Hz", "50Hz"])
    y = np.array([0.18834297, 1.98414867e-01, 0.49548176, 0.66411244, 0.66245299])

    ax8 = plt.subplot(2, 4, 8)
    ax8.bar(x, y)
    ax8.set_yticklabels([])
    ax8.set_ylim([0, 1])
    ax8.set_title('F1 Walking')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.show()'''

    ''''# Overall best results
    acc = np.array([0.17731791, 0.55602312, 0.93278539, 0.72834384, 0.90076908, 0.57946871, 0.77698193, 0.66411244])
    act = np.array(["Climbing-up 50Hz", "Climbing-down 25Hz ", "Jumping 50Hz ", "Lying 25Hz ", "Running 50Hz ",
           "Sitting 3.125Hz ", "Standing 25Hz ", "Walking 25Hz "])
    c = ['red', 'green', 'red', 'green', 'red', 'blue', 'green', 'green']
    plt.bar(act, height=acc, color = c)
    plt.title("Best F1 scores for all 8 activities for RWHAR dataset")
    plt.xlabel("Activity")
    plt.ylabel("Accuracy")
    plt.show()'''

    '''act = np.array(
        ["Climbing-up", "Climbing-down", "Jumping", "Lying", "Running", "Sitting", "Standing", "Walking", "Total"])
    savings = np.array([0.0, 50.0, 0.0, 50.0, 0.0, 62.5, 50.0, 50.0, 262.5])
    c = ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'green']
    plt.bar(act, savings, color=c)
    plt.title("Total saving by picking best accuracy")
    plt.xlabel("Activity")
    plt.ylabel("Savings")
    plt.show()'''

    with open('test_gt.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))

    data = sum(data, [])
    print(data)

def plot(acc, prec, rec, f1):
    print("Inside Plot functionality")
    # create data
    l1 = acc
    l2 = prec
    l3 = rec
    l4 = f1
    # plot lines
    plt.plot(l1, l2, label="Accuracy vs Precision")
    plt.plot(l3, l4, label="Recall vs F1")
    plt.legend()
    plt.show()

def final_plot():
    print("Inside final plot")


if __name__ == '__main__':
    print("Inside main function")
    # data = load_data()
    # resampling(data, FACTOR)
    #evaluation()
    final_plot()