import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/tbacha/DMML2019_Team_Apple/master/data/ks-projects-201801.csv")

# dtypes
df["deadline"] = pd.to_datetime(df["deadline"])
df["launched"] = pd.to_datetime(df["launched"])

# states
criteria = df["state"].isin(["failed", "successful", "canceled"])
df = df[criteria]
df.loc[(df["state"] == "canceled") & (df["usd_pledged_real"] >= df["usd_goal_real"]), "state"] = "successful"
df.loc[(df["state"] == "canceled") & (df["usd_pledged_real"] < df["usd_goal_real"]), "state"] = "failed"
df.loc[(df["state"] == "successful"), "state"] = 1
df.loc[(df["state"] == "failed"), "state"] = 0

# other features
df = df.drop(["goal", "pledged", "usd pledged"], axis=1)
df["pct_reached"] = round(df["usd_pledged_real"] / df["usd_goal_real"], 2)
df["elapsed_time"] = df["deadline"] - df["launched"]

# noise
criteria = (((df["backers"]==0) & (df["usd_pledged_real"] > 0) | (df["country"] == 'N,0"')))
df = df[~criteria]

# index
df.reset_index(inplace=True, drop=True)