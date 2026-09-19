import sklearn
import pandas as pd

data = sklearn.datasets.fetch_openml("credit-g", version=1, as_frame=True)

df = data.frame

df_shape = df.shape
df_types = df.dtypes

counts_value = df["class"].value_counts(normalize=True)

print(f"DataFrame shape: {df_shape}")
print(f"DataFrame types:\n{df_types}")
print(f"Class distribution:\n{counts_value}")

print(data.DESCR)
df.to_csv("data/raw.csv", index=False)





#df.to_csv("data/raw.csv", index=False)