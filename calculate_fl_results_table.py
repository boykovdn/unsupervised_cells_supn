import pickle
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE = "./1603_exp_dataframes_resto_sampled.pickle"
bar_width = 0.3
y_range = [0.6, 0.9]
ticks_formatted = {
        'resto_supn' : "SUPN\n(Restoration)",
        'resto_diag' : "Diag\n(Restoration)",
        'resto_l2' : "L2\n(Restoration)",
        'supn' : "SUPN",
        'diag' : "Diag",
        'l2' : 'L2'
        }
xticks_fontsize = 'x-small'
yticks_fontsize = 'small'

dfs = None
with open(INPUT_FILE, "rb") as fin:
    dfs = pickle.load(fin)

model_names = dfs[0]['model_name']
aurocs = []
auprcs = []
for df in dfs:
    aurocs.append(torch.Tensor(df['auroc']))
    auprcs.append(torch.Tensor(df['auprc']))

aurocs = torch.stack(aurocs, dim=0)
auprcs = torch.stack(auprcs, dim=0)

print(model_names)
print(aurocs.mean(0))
print(aurocs.std(0))

x_ticks = np.array(range(len(model_names)))
fig, ax = plt.subplots()
ax.bar(x_ticks, aurocs.mean(0), bar_width, yerr=aurocs.std(0), label="AUROC")
ax.bar(x_ticks + bar_width, auprcs.mean(0), bar_width, yerr=auprcs.std(0), label="AUPRC")
ax.set_ylim(y_range)
ax.set_xticks(x_ticks + bar_width/2, labels=list(map(lambda n : ticks_formatted[n], model_names)), 
        fontsize=xticks_fontsize)
plt.tick_params(axis='y', which='major', labelsize=yticks_fontsize)
ax.legend()
plt.show()
