import pandas as pd
import numpy as np

def _draw_one_pot(pot_df, A, B, rng):
    """Return one randomized Treatment Series for this pot under the rules."""
    pot_df = pot_df.copy()
    pid = pot_df["Pot ID"].iloc[0]
    try:
        pid_int = int(str(pid).strip())
    except Exception:
        pid_int = pid  # if not int-like, won't match {1..6}

    t_new = pot_df["Treatment"].copy()

    idxA = pot_df.index[pot_df["Treatment"].isin(A)]
    idxB = pot_df.index[pot_df["Treatment"].isin(B)]
    A_labels = pot_df.loc[idxA, "Treatment"].to_numpy()
    B_labels = pot_df.loc[idxB, "Treatment"].to_numpy()

    if pid_int in {1, 2, 3}:
        # Shuffle labels within each group independently
        if len(idxA) > 1:
            rng.shuffle(A_labels)
            t_new.loc[idxA] = A_labels
        if len(idxB) > 1:
            rng.shuffle(B_labels)
            t_new.loc[idxB] = B_labels
        return t_new

    if pid_int in {4, 5, 6}:
        # Force exactly 4 T3 and 4 T4 within B
        uniq = pd.unique(B_labels)
        assert set(uniq) >= {"T3", "T4"}, f"Pot {pid}: B must contain T3 and T4, got {list(uniq)}"
        m = len(idxB)
        assert m == 8, f"Pot {pid}: expected 8 B items, got {m}"
        # Choose 4 positions for T3 uniformly at random, rest T4
        idxB_list = np.array(idxB)
        chosen = rng.choice(idxB_list, size=4, replace=False)
        t_new.loc[idxB] = "T4"
        t_new.loc[chosen] = "T3"
        # A stays as originally labeled
        return t_new

    # Fallback: just shuffle within groups (same as pots 1–3)
    if len(idxA) > 1:
        rng.shuffle(A_labels); t_new.loc[idxA] = A_labels
    if len(idxB) > 1:
        rng.shuffle(B_labels); t_new.loc[idxB] = B_labels
    return t_new


def build_permutation_random(df, A, B, treatment, n_perm=5000, seed=None):
    """
    Generate n_perm global permutations consistent with the per-pot rules.
    Returns a long DF with columns: Treatment_perm, treatment, perm_id, ...
    """
    rng = np.random.default_rng(seed)
    frames = []
    # Work with pots in a stable order
    pot_groups = [g for _, g in df.groupby("Pot ID", sort=True)]

    for perm_id in range(n_perm):
        # draw one randomized assignment per pot, then combine
        draws = [_draw_one_pot(g, A, B, rng) for g in pot_groups]
        combined = pd.concat(draws).sort_index()
        tmp = df.copy()
        tmp["Treatment_perm"] = combined
        tmp["treatment"] = combined.map(lambda t: "T" if t in treatment else "C")
        tmp["perm_id"] = perm_id
        frames.append(tmp)

    return pd.concat(frames, ignore_index=True)


def perm_test_rand(df, A, B, treatment, metric, agg_fn, n_perm = 10_000, seed = 17):
    #example inputs:
    # A = {"T1", "T3"}
    # B = {"T2", "T4"}
    # treatment = {"T1", "T2"}
    # metric = "Average Score" - col names
    # agg_fn = "mean" - mean, median 
    permuted_long = build_permutation_random(df, A, B, treatment, n_perm=n_perm, seed=seed)
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
    
    
    # mark treatment vs control
    df["treatment"] = df["Treatment"].map(lambda t: "T" if t in treatment else "C")
    
    # aggregate within each treatment group
    g_obs = (
        df.groupby("treatment", as_index=False)[metric]
        .mean()   # or .agg('mean') if you want to keep same syntax
    )
    
    obs_stat = (
                    g_obs.loc[g_obs["treatment"] == "T", metric].iloc[0]
                    - g_obs.loc[g_obs["treatment"] == "C", metric].iloc[0]
    )

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

# import pandas as pd

# exp1_data = pd.read_csv("data/processed_data/exp1_clean.txt", sep=",")
# exp12_data = pd.read_csv("data/processed_data/exp12_comb.csv")
# exp12_data_photo =pd.read_csv("data/processed_data/exp12_comb_photo.csv")
# exp1_data_photo = exp12_data_photo[exp12_data_photo["Pot ID"].isin(["P1", "P2", "P3"])]

# A = {"T1", "T2"}
# B = {"T3", "T4"}
# treatment = {"T1", "T3"}


# metric = "Photo Freshness Delta"
# agg_fn = "mean"

# #perm_data, p_left, p_two, p_right, obs_stat = perm_test(exp12_data_photo, A, B, treatment, metric, agg_fn)
# perm_data, p_left, p_two, p_right, obs_stat = perm_test_rand(exp12_data_photo, A, B, treatment, metric, agg_fn, n_perm=3, seed=17)
# print(obs_stat)