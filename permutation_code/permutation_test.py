import pandas as pd
import numpy as np
from itertools import product


def pot_perms(pot_df, A, B):
    """Return the 4 treatment-label permutations for one pot (keep/swap in A and in B)."""
    pot_df = pot_df.copy()
    idxA = pot_df.index[pot_df["Treatment"].isin(A)]
    idxB = pot_df.index[pot_df["Treatment"].isin(B)]

    # original order within each group (row order)
    A_labels = list(pot_df.loc[idxA, "Treatment"])
    B_labels = list(pot_df.loc[idxB, "Treatment"])

    A_opts = [A_labels, A_labels[::-1]]   # keep or swap
    B_opts = [B_labels, B_labels[::-1]]   # keep or swap

    out = []
    for a_choice, b_choice in product(A_opts, B_opts):
        t_new = pot_df["Treatment"].copy()
        for j, ix in enumerate(idxA):
            t_new.loc[ix] = a_choice[j]
        for j, ix in enumerate(idxB):
            t_new.loc[ix] = b_choice[j]
        out.append(t_new)
    return out  # list of Series, each aligned to pot_df.index


def build_permutation(df, A, B, treatment):
    # get the 4 options per pot (3 pots -> 4*4*4 = 64)
    per_pot = [pot_perms(g, A, B) for _, g in df.groupby("Pot ID", sort=True)]

    # cartesian product across pots to build global permutations
    frames = []
    perm_id = 0
    for choices in product(*per_pot):
        tmp = df.copy()
        combined = pd.concat(choices).sort_index()
        tmp["treatment"] = combined.map(lambda t: "T" if t in treatment else "C")
        tmp["Treatment_perm"] = combined
        tmp["perm_id"] = perm_id
        frames.append(tmp)
        perm_id += 1

    permuted_long = pd.concat(frames, ignore_index=True)
    return permuted_long


def perm_test(df, A, B, treatment, metric, agg_fn):
    #example inputs:
    # A = {"T1", "T3"}
    # B = {"T2", "T4"}
    # treatment = {"T1", "T2"}
    # metric = "Average Score" - col names
    # agg_fn = "mean" - mean, median 
    permuted_long = build_permutation(df, A, B, treatment)
    g = (
        permuted_long
        .groupby(["perm_id", "treatment"], as_index=False)[metric]
        #.mean()
        .agg(agg_fn)
        .reset_index()
    )

    stats = (
        g.pivot(index="perm_id", columns="treatment", values=metric)  # 64×2 table
        .reindex(columns=["T","C"])                                  # enforce order
        .assign(diff=lambda d: d["T"] - d["C"])                      # T - C
        .reset_index()[["perm_id","diff"]]
    )
    
    obs_stat = stats[stats["perm_id"] == 0]


    obs_stat = float(stats.loc[stats["perm_id"] == 0, "diff"])
    perm = stats.loc[stats["perm_id"] != 0, "diff"].to_numpy()


    n = perm.size
    p_right = np.sum(perm >= obs_stat) / n   # H1: T > C
    p_left  = np.sum(perm <= obs_stat) / n    # H1: T < C
    p_two   = np.sum(np.abs(perm) >= abs(obs_stat)) / n 

    #BF correction
    p_right = min(1, p_right*2)
    p_left = min(1, p_left*2)
    p_two = min(1, p_two*2)

    return stats, p_left, p_two, p_right, obs_stat



#df = pd.read_csv("data/processed_data/exp1_clean.txt", sep=",")
df = pd.read_csv("data/processed_data/exp12_comb.csv")

#fridge
# A = {"T1", "T3"}
# B = {"T2", "T4"}
# treatment = {"T1", "T2"}
A = {"T1", "T2"}
B = {"T3", "T4"}
treatment = {"T1", "T3"}


metric = "Average Score"
agg_fn = "mean"

