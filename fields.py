import deeplake

# Load the DeepLake dataset
ds = deeplake.load('hub://activeloop/wiki-art')

# Display dataset information using built-in methods
print("Dataset Information:")
ds.info(verbose=True)

# Alternatively, print number of samples and available fields
num_samples = ds.summary()['total_samples']
fields = list(ds.tensors.keys())
print(f"\nNumber of samples: {num_samples}")
print(f"Fields in the dataset: {', '.join(fields)}")

# Use DeepLake's 'dataset.sample()' method to retrieve random samples
samples = ds.sample(n=5, seed=42)  # Get 5 random samples

# Display the retrieved samples
for idx, sample in enumerate(samples):
    print(f"\nSample {idx + 1}:")
    for key, value in sample.items():
        print(f"  {key}: {value}")
