import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("./EX2/data/result.csv").sort_values(by="pred",ascending=False).reset_index(drop=True)

data["tp_cumsum"] = (data["label"] == 1).cumsum()
data["fp_cumsum"] = (data["label"] == 0).cumsum()

total_pos = (data["label"] == 1).sum()

data["precision"] = data["tp_cumsum"] / (data["tp_cumsum"] + data["fp_cumsum"])
data["recall"] = data["tp_cumsum"] / total_pos

recall = np.r_[0.0, data["recall"].to_numpy()]
precision = np.r_[1.0, data["precision"].to_numpy()]

# 计算 AUPR（recall 单调增时可用梯形法则）
aupr = np.trapz(precision, recall)

plt.figure()
plt.step(recall, precision, where="post")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall Curve (AUPR = {aupr:.4f})")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.savefig("./EX2/PR_curve.png")
plt.show()