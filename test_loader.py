from dataset_loader import load_dataset

X, y = load_dataset()

print("Total images:", X.shape)
print("Total labels:", y.shape)
print("Real:", sum(y))
print("Spoof:", len(y) - sum(y))
