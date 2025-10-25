import numpy as np

def randomize_order(plant_df, rng):
    return rng.permutation(plant_df)

with open('exp2_water_assignments.txt', 'r') as file:
    # Read all lines into a list
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    lines = np.array(lines)

rng = np.random.default_rng(seed=914923)
shuffled = randomize_order(lines, rng)
#print(shuffled)

# For experiment 1 smell test
pots = [1]*4 + [2]*4 + [3]*4
pots = np.reshape(pots, (12,1))
treatments = ["T1", "T2", "T3", "T4"]*3
treatments = np.reshape(treatments, (12,1))
plant_df = np.hstack((pots, treatments))
exp1_rng = np.random.default_rng(seed=523820)
exp1_shuffled = randomize_order(plant_df, exp1_rng)
print(exp1_shuffled)