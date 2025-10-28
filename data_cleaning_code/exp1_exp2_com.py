import pandas as pd

df_exp1 =  pd.read_csv("data/processed_data/exp1_clean.txt")
#filtered_1 = df_exp1[df_exp1["Treatment"].isin(["T3", "T4"])][["Pot ID", "Treatment", "Initial Weights", "Day 5 Weight", "Average Score"]]
df_exp1 = df_exp1[["Pot ID", "Treatment", "Initial Weights", "Day 5 Weight", "Average Score", "Weight Delta"]]

df_exp2 =  pd.read_csv("data/processed_data/exp2_clean.txt")
#filtered_2 = df_exp2[df_exp2["Treatment"].isin(["T3", "T4"])][["Pot ID", "Treatment","Initial Weights", "Day 5 Weight","Average Score"]]
df_exp2 = df_exp2[["Pot ID", "Treatment","Initial Weights", "Day 5 Weight","Average Score", "Weight Delta"]]



df_out = pd.concat([df_exp1, df_exp2], ignore_index=True)

df_out['Ranked Average Score'] = df_out['Average Score'].rank()


df_out.to_csv("data/processed_data/exp12_comb.csv", index=False)