import numpy as np

data = np.load("matrix_N95/data.npy")
time_index = np.load("matrix_N95/time_index.npy", allow_pickle=True)
met = np.load("matrix_N95/met_raw_aligned_cache.npz")

print("data.npy shape:", data.shape)
print("time_index.npy shape:", time_index.shape)
print("time range:", time_index[0], "to", time_index[-1])

print("met keys:", met.files)
for k in met.files:
    print(k, met[k].shape)

print("O3 missing count:", np.isnan(data).sum())
print("O3 missing ratio:", np.isnan(data).sum() / data.size)