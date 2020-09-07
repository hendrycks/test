import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from calib_tools import rmsce

k = 5
base_dir = "results"
models = [f.split("results_")[1] for f in os.listdir(base_dir) if "results" in f]
print(models)

for model in models:
    dir = os.path.join(base_dir, "results_{}".format(model))
    fnames = [f for f in os.listdir(dir) if ".csv" in f]

    all_max_probs = []
    all_cors = []
    all_accs = []
    all_confs = []

    for fname in fnames:
        subject = fname.split(".csv")[0]
        fpath = os.path.join(dir, fname)
        df = pd.read_csv(fpath)

        max_probs = []
        cors = []
        for i in range(df.shape[0]):
            probs = [df["{}_choice{}_probs".format(model, choice)][i] for choice in ["A", "B", "C", "D"]]
            cors.append(int(df["{}_correct".format(model)][i]))
            max_probs.append(np.max(probs))
        all_max_probs += max_probs
        all_cors += cors
        all_accs.append(np.mean(cors))
        all_confs.append(np.mean(max_probs))

    avg_max_prob = np.mean(all_max_probs)
    acc = np.mean(all_cors)
    rms_ce = rmsce(np.array(all_cors), np.array(all_max_probs))
    print("{} overall conf: {:.3f}, acc: {:.3f}, RMS: {:.3f}".format(model, avg_max_prob, acc, rms_ce))

    plt.scatter(all_confs, all_accs)
    min = np.minimum(np.min(all_confs), np.min(all_accs))
    max = np.maximum(np.max(all_confs), np.max(all_accs))
    x = np.arange(min, max, 0.01)
    y = np.arange(min, max, 0.01)
    plt.plot(x, y, c="r")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.savefig("{}_calibration".format(model))

