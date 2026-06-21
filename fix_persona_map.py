"""
fix_persona_map.py
Run once to patch segwise_model.pkl with meaningful persona names
based on which cluster has highest average Total_Spending.
"""
import pickle, pandas as pd

# Load current bundle
with open("models/segwise_model.pkl", "rb") as f:
    bundle = pickle.load(f)

# Load predictions to get spending per cluster
df = pd.read_csv("outputs/cluster_predictions.csv")

# Rank clusters by mean Total_Spending (descending)
spend_rank = (
    df.groupby("Cluster")["Total_Spending"]
    .mean()
    .rank(ascending=False)
    .astype(int)
)

rank_to_name = {1: "VIP Shoppers", 2: "Deal Hunters",
                3: "Casual Buyers", 4: "Dormant Users"}

new_persona_map = {int(cid): rank_to_name[int(rank)]
                   for cid, rank in spend_rank.items()}

print("Old persona_map:", bundle["persona_map"])
print("New persona_map:", new_persona_map)

# Patch bundle
bundle["persona_map"] = new_persona_map

# Save
with open("models/segwise_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

# Patch the CSV too
df["Persona"] = df["Cluster"].map(new_persona_map)
df.to_csv("outputs/cluster_predictions.csv", index=False)

print("Done. PKL and CSV updated with real persona names.")
print("Persona distribution:")
print(df["Persona"].value_counts())
