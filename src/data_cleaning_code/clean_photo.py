import pandas as pd

# import csv
df = pd.read_csv("data/raw_data/exp1.csv")

df = df[["image_id", "freshness_index"]]

df[["Pot ID", "Treatment", "Day"]] = (
    df["image_id"]
    .apply(lambda x: [x[:2], x[2:4], x[5:]])  # adjust positions
    .apply(pd.Series)
)
df = df.drop(columns=["image_id"])
df = df[df["Day"].isin(["1", "5"])]

delta = (
    df.pivot_table(
        index=["Pot ID", "Treatment"],
        columns="Day",
        values="freshness_index"
    )
    .rename_axis(columns=None)                # <- strips the "Day" label
    .assign(delta=lambda x: x["5"] - x["1"])  # Day5 − Day1
    .dropna(subset=["1","5"])
    .reset_index()[["Pot ID", "Treatment", "delta"]]
)

delta_exp1 = delta


df_2 = pd.read_csv("data/raw_data/all_metrics_final.csv")
df_2 = df_2[["ID", "freshness_index"]]

df_2[["Pot ID", "Plant ID", "Day"]] = (
    df_2["ID"]
    .apply(lambda x: [x[16:17],x[25:26],x[9:10]])  # adjust positions
    .apply(pd.Series)
)
df_2 = df_2.drop(columns=["ID"])
df_2 = df_2[df_2["Day"].isin(["1", "5"])]

df_map_treat = pd.read_csv("data/raw_data/exp2_smell_test_results.txt")
df_map_treat = df_map_treat[["Pot ID", "Plant ID", "Treatment"]]

#mapping to str so we can merge 
df_2["Pot ID"] = df_2["Pot ID"].astype(str)
df_2["Plant ID"] = df_2["Plant ID"].astype(str)

df_map_treat["Pot ID"] = df_map_treat["Pot ID"].astype(str)
df_map_treat["Plant ID"] = df_map_treat["Plant ID"].astype(str)

df_2 = df_2.merge(df_map_treat[["Pot ID", "Plant ID", "Treatment"]],
                on=["Pot ID", "Plant ID"], how="left")

df_2["Pot ID"] = "P" + df_2["Pot ID"].astype(str)

delta_2 = (
    df_2.pivot_table(
        index=["Pot ID", "Plant ID", "Treatment"],
        columns="Day",
        values="freshness_index"
    )
    .rename_axis(columns=None)                # <- strips the "Day" label
    .assign(delta=lambda x: x["5"] - x["1"])  # Day5 − Day1
    .dropna(subset=["1","5"])
    .reset_index()[["Pot ID", "Plant ID", "Treatment", "delta"]]
)

delta_2 = delta_2.drop(columns=["Plant ID"])

delta_exp2 = delta_2

#combined photo metric
df_photo_comb_exp12 = pd.concat([delta_exp1, delta_exp2], ignore_index=True)
df_photo_comb_exp12 = df_photo_comb_exp12.rename(columns={"delta": "Photo Freshness Delta"})

#can't easily combine with other dataset
df_photo_comb_exp12.to_csv("data/processed_data/exp12_comb_photo.csv", index=False)