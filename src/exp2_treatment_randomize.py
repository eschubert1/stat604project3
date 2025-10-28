# Randomize water treatment to plants
# There are 3 pots of plants, each of which had 8 stems harvested from them.
# For each pot, 4 of the stems get assigned to water, 4 do not

import numpy as np

# Set seed
np.random.seed(seed=353754)

pot_id = [4]*8 + [5]*8 + [6]*8
plant_id = list(range(1,9))*3

exp2_df = np.array([pot_id, plant_id])
exp2_df = exp2_df.T

def assign_water_treatments(df):
    pot_ids = np.unique(df[:,0])
    assignments = np.zeros(np.shape(df)[0])
    for i in pot_ids:
        # Choose subset of indices
        ii = np.where(df[:,0]==i)
        ii = np.reshape(ii, 8)
        # Choose random sample of size 4 for water treatments
        water = np.random.choice(ii, 4, replace=False)
        assignments[water] = 1
    assignments = np.reshape(assignments, (24, 1))
    df_assigned = np.hstack((df, assignments))
    print(df_assigned)

assign_water_treatments(exp2_df)
