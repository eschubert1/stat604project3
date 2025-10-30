import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

exp1_images = pd.read_csv("data/processed_data/exp1.csv")
exp2_images = pd.read_csv("data/processed_data/all_metrics_final.csv")
exp2_treatments = pd.read_csv("data/processed_data/exp2_clean.txt", sep=',')

exp1_images["Pot ID"] = 0
exp1_images["Treatment"] = ""
exp1_images["Day"] = 0

exp2_images["Pot ID"] = 0
exp2_images["Plant ID"] = 0
exp2_images["Treatment"] = ""
exp2_images["Day"] = 0

# Process experiment 1 treatments
for i in range(np.shape(exp1_images)[0]):
    image_name = exp1_images.iloc[i,0]
    nums=[]
    for char in image_name:
        if char.isdigit():
            nums.append(int(char))
    if nums[1] == 0:
        continue
    else:
        exp1_images.loc[i, "Pot ID"] = nums[0]
        exp1_images.loc[i, "Treatment"] = "T" + str(nums[1])
        exp1_images.loc[i, "Day"] = nums[2]

# Process experiment 2 treatments
trt_col = np.shape(exp2_images)[1]-2
for i in range(np.shape(exp2_images)[0]):
    image_name = exp2_images.iloc[i,0]
    nums = []
    for char in image_name:
        if char.isdigit():
            nums.append(int(char))
    exp2_images.loc[i, "Pot ID"] = nums[2]
    exp2_images.loc[i, "Plant ID"] = nums[3]
    a = np.equal(exp2_treatments["Pot ID"],nums[2])
    b = np.equal(exp2_treatments["Plant ID"], nums[3])
    ii = np.where(a & b)[0]
    #ii = np.where(np.equal(exp2_treatments["Pot ID"],nums[2]) & 
    #              np.equal(exp2_treatments["Plant ID"], nums[3]))
    #print(ii)
    exp2_images.iloc[i, trt_col] = exp2_treatments.iloc[ii, 4].values[0]
    #print(exp2_images.loc[i, "Treatment"])
    #print(exp2_treatments.loc[ii, "Treatment"])
    exp2_images.loc[i, "Day"] = nums[1]


exp1_image_freshness = exp1_images[["Pot ID", "Treatment", "Day", "freshness_index"]]
exp2_image_freshness = exp2_images[["Pot ID", "Treatment", "Day", "freshness_index"]]
freshness_results = pd.concat([exp1_image_freshness, exp2_image_freshness], ignore_index=True)
freshness_results = freshness_results[freshness_results['Treatment']!='']
freshness_results.Treatment = freshness_results.Treatment.apply(str)
freshness_results.Day = freshness_results.Day.apply(int)
freshness_results.freshness_index = freshness_results.freshness_index.apply(float)

def image_freshness_over_time(df, path):

    groups = df.groupby(['Treatment', 'Day'])['freshness_index'].mean().reset_index()
    groups = groups.groupby(['Treatment'])
    trt_labels = ['F+W', 'F+NW', 'NF+W', 'NF+NW']
    for name, group in groups:
    #    #fresh = group['freshness_index'].mean()
        plt.plot(group.Day, group.freshness_index, marker='o', linestyle='--', markersize=12, label=name)

    #plt.plot(groups.Day, groups.freshness_index, marker='o', linestyle='--', markersize=12, label=groups.Treatment)

    plt.legend(labels = trt_labels)
    plt.xlabel('Day in Experiment')
    plt.ylabel('Average Image Freshness Index')

    plt.savefig(path)
    plt.clf()
    

image_freshness_over_time(freshness_results, "figures/image_freshness.pdf")
image_freshness_over_time(freshness_results, "figures/image_freshness.png")

print(freshness_results.Treatment.unique())