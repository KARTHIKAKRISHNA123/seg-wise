import pickle

with open("models/segwise_model.pkl", "rb") as f:
    bundle = pickle.load(f)

print("Keys:", list(bundle.keys()))
print("n_clusters:", bundle.get("n_clusters"))
print("best_name:", bundle.get("best_name"))
print("persona_map:", bundle.get("persona_map"))
print("features:", bundle.get("features"))
