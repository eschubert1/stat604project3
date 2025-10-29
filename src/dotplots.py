import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

def make_exp1_dataframe(path):
    exp_test = []
    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            exp_test.append(row)

    exp_test = np.array(exp_test)
    exp_results = np.array(exp_test[1:,:])
    exp_results = pd.DataFrame({
        'Pot ID' : exp_results[:,0],
        'Treatment' : exp_results[:,1],
        'Day 5 Weight' : exp_results[:,2],
        'Test Order' : exp_results[:,3],
        'Filippo\'s Score' : [float(score) for score in exp_results[:,5]],
        'Josh\'s Score' : [float(score) for score in exp_results[:,6]]
    })
    exp_results['score'] = (exp_results['Filippo\'s Score'] + exp_results['Josh\'s Score'])/2
    
    return exp_results

def make_exp2_dataframe(path):
    exp_test = []
    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            exp_test.append(row)

    exp_test = np.array(exp_test)
    exp_results = np.array(exp_test[1:,:])
    exp_results = pd.DataFrame({
        'Pot ID' : exp_results[:,0],
        'Plant ID' : exp_results[:, 1],
        'Weight (Day 0)' : exp_results[:,2],
        'Water treatment' : exp_results[:, 3],
        'Treatment' : exp_results[:,4],
        'Weight (Day 5)' : exp_results[:,5],
        'Had Mold' : exp_results[:, 6],
        'Smell test intended order' : exp_results[:, 7],
        'Smell test actual order' : exp_results[:,8],
        'Josh\'s Score' : [float(score) for score in exp_results[:,9]],
        'Mihir\'s Score' : [float(score) for score in exp_results[:,10]],
        'Filippo\'s Score' : [float(score) for score in exp_results[:,11]]
    })
    exp_results['score'] = (exp_results['Filippo\'s Score'] +
                            exp_results['Mihir\'s Score'] +
                            exp_results['Josh\'s Score'])/3
    
    return exp_results

def plot_scores(df, path):

    groups = df.groupby('Treatment')
    trt_labels = ['F+W', 'F+NW', 'NF+W', 'NF+NW']
    for name, group in groups:
        plt.plot(group.Treatment, group.score, marker='o', linestyle='', markersize=12, label=name)

    plt.legend(labels = trt_labels)
    plt.xlabel('Treatment')
    plt.xticks(ticks = ["T1", "T2", "T3", "T4"], labels=trt_labels)
    plt.ylabel('Average Freshness Score')

    plt.savefig(path)
    plt.clf()


exp1_results = make_exp1_dataframe('data/raw_data/exp1_results.txt')
exp2_results = make_exp2_dataframe('data/raw_data/exp2_smell_test_results.txt')
combined_results = pd.concat([exp1_results[['Treatment', 'score']], 
                              exp2_results[['Treatment', 'score']]], ignore_index=True)

plot_scores(combined_results, 'figures/average_freshness_all.pdf')
plot_scores(combined_results, 'figures/average_freshness_all.png')
plot_scores(exp1_results, "figures/average_freshness_exp1.pdf")
plot_scores(exp2_results, "figures/average_freshness_exp2.pdf")