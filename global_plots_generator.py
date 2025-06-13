import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils

offset = 0

sns.set(font_scale=1.2)
sns.set_style("whitegrid")
df = pd.read_csv('summary/summarized_data.csv', index_col=['ViT Name'])
original_acc = df.filter(regex="ViT-Defa*", axis=0)["Accuracy"].item()/100
original_time = df.filter(regex="ViT-Defa*", axis=0)["Total Latency"].item()
original_mem = df.filter(regex="ViT-Defa*", axis=0)["Max Mem use"].item()

prunning_data = df.filter(regex='ViT-Pru*', axis=0)
#prunning_data["Accuracy"] = 1 - prunning_data["Accuracy"]
prunning_data["Pruning"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
prunning_data["Mem Latency"] = 0.24
#prunning_data["Max Mem use"] = 344.3725
prunning_data = prunning_data.sort_values("Pruning", ascending=True)

utils.plot_metrics_line(prunning_data, "Pruning", "Metrics against pruning", save=True, save_name="Pruning", sub_folder="Pruning")

prunning_data["Accuracy"] = prunning_data["Accuracy"]/100
diff_latency = original_time - prunning_data["Total Latency"]
diff_mem = original_mem - prunning_data["Max Mem use"]
diff_acc = original_acc - prunning_data["Accuracy"]

tradeoff_latency = (diff_latency/original_time)/(diff_acc/original_acc + offset) * (prunning_data["Accuracy"]**2)
tradeoff_mem = (diff_mem/original_mem)/(diff_acc/original_acc + offset) * (prunning_data["Accuracy"]**2)

prunning_data["Values time"] = tradeoff_latency
prunning_data["Values mem"] = tradeoff_mem

utils.plot_elasticities_line(prunning_data, "Pruning", "Elasticities against pruning", save=True, save_name="Pruning", sub_folder="Pruning")

prunning_data = df.filter(regex='ViT-Mer*', axis=0)
#prunning_data["Accuracy"] = 1 - prunning_data["Accuracy"]
prunning_data["Pruning"] = [10,15,20,25,30,5]
prunning_data["Mem Latency"] = 0.24
prunning_data["Max Mem use"] = 344.3725
prunning_data = prunning_data.sort_values("Pruning", ascending=True)

utils.plot_metrics_line(prunning_data, "r", "Metrics against merging", save=True, save_name="Merging", sub_folder="Merging")

prunning_data["Accuracy"] = prunning_data["Accuracy"]/100
diff_latency = original_time - prunning_data["Total Latency"]
diff_mem = original_mem - prunning_data["Max Mem use"]
diff_acc = original_acc - prunning_data["Accuracy"]

tradeoff_latency = (diff_latency/original_time)/(diff_acc/original_acc + offset) * (prunning_data["Accuracy"]**2)
tradeoff_mem = (diff_mem/original_mem)/(diff_acc/original_acc + offset) * (prunning_data["Accuracy"]**2)

prunning_data["Values time"] = tradeoff_latency
prunning_data["Values mem"] = tradeoff_mem

utils.plot_elasticities_line(prunning_data, "r", "Elasticities against merging",save=True, save_name="Merging", sub_folder="Merging")

"""

Begining of quantized + technique

"""

prunning_data = df.filter(regex='ViT-Combine-Pr(?!.*Merging).*$', axis=0)
#prunning_data["Accuracy"] = 1 - prunning_data["Accuracy"]
prunning_data["Pruning"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
prunning_data = prunning_data.sort_values("Pruning", ascending=True)

utils.plot_metrics_line(prunning_data, "Pruning", "Metrics against pruning", save=True, save_name="PruningQuant", sub_folder="Combine/PQuant")

prunning_data["Accuracy"] = prunning_data["Accuracy"]/100
diff_latency = original_time - prunning_data["Total Latency"]
diff_mem = original_mem - prunning_data["Max Mem use"]
diff_acc = original_acc - prunning_data["Accuracy"]

tradeoff_latency = (diff_latency/original_time)/(diff_acc/original_acc + offset) * (prunning_data["Accuracy"]**2)
tradeoff_mem = (diff_mem/original_mem)/(diff_acc/original_acc + offset) * (prunning_data["Accuracy"]**2)

prunning_data["Values time"] = tradeoff_latency
prunning_data["Values mem"] = tradeoff_mem

utils.plot_elasticities_line(prunning_data, "Pruning", "Elasticities against pruning", save=True, save_name="PruningQuant", sub_folder="Combine/PQuant")

prunning_data = df.filter(regex='ViT-Combine-Mer*', axis=0)
prunning_data["Pruning"] = [10,15,20,25,30,5]
prunning_data = prunning_data.sort_values("Pruning", ascending=True)

utils.plot_metrics_line(prunning_data, "r", "Metrics against merging", save=True, save_name="MergingQuant", sub_folder="Combine/MQuant")

prunning_data["Accuracy"] = prunning_data["Accuracy"]/100
diff_latency = original_time - prunning_data["Total Latency"]
diff_mem = original_mem - prunning_data["Max Mem use"]
diff_acc = original_acc - prunning_data["Accuracy"]

tradeoff_latency = (diff_latency/original_time)/(diff_acc/original_acc + offset) * (prunning_data["Accuracy"]**2)
tradeoff_mem = (diff_mem/original_mem)/(diff_acc/original_acc + offset) * (prunning_data["Accuracy"]**2)

prunning_data["Values time"] = tradeoff_latency
prunning_data["Values mem"] = tradeoff_mem

utils.plot_elasticities_line(prunning_data, "r", "Elasticities against merging",save=True, save_name="MergingQuant", sub_folder="Combine/MQuant")

"""

Begining of all techniques combos

"""

merging = [10, 15, 20, 25, 30, 5] * 8
pruning = [0.1]*6 + [0.2]*6 + [0.3]*6 + [0.4]*6 + [0.5]*6 + [0.6]*6 + [0.7]*6 + [0.8]*6

combined = df.filter(regex='ViT-Combine-Pr(?!.*Quantized).*$', axis=0)
combined["Pruning"] = pruning
combined["Merging"] = merging

utils.plot_metrics_heat(combined, "Metrics heatmaps", save=True, save_name="PM", sub_folder="Combine/PM")

combined["Accuracy"] = combined["Accuracy"]/100
diff_latency = original_time - combined["Total Latency"]
diff_mem = original_mem - combined["Max Mem use"]
diff_acc = original_acc - combined["Accuracy"]

tradeoff_latency = (diff_latency/original_time)/(diff_acc/original_acc + offset) * (combined["Accuracy"]**2)
tradeoff_mem = (diff_mem/original_mem)/(diff_acc/original_acc + offset) * (combined["Accuracy"]**2)

combined["Values time"] = tradeoff_latency
combined["Values mem"] = tradeoff_mem

utils.plot_elasticities_heat(combined, "Elasticity heatmaps", save=True, save_name="PM", sub_folder="Combine/PM")


combined = df.filter(regex='ViT-Combine-Pr.*Mer.*Quantized$', axis=0)
combined["Pruning"] = pruning
combined["Merging"] = merging

utils.plot_metrics_heat(combined, "Metrics heatmaps", save=True, save_name="PMQuant", sub_folder="Combine/PMQuant")

combined["Accuracy"] = combined["Accuracy"]/100
diff_latency = original_time - combined["Total Latency"]
diff_mem = original_mem - combined["Max Mem use"]
diff_acc = original_acc - combined["Accuracy"]

tradeoff_latency = (diff_latency/original_time)/(diff_acc/original_acc + offset) * (combined["Accuracy"]**2)
tradeoff_mem = (diff_mem/original_mem)/(diff_acc/original_acc + offset) * (combined["Accuracy"]**2)

combined["Values time"] = tradeoff_latency
combined["Values mem"] = tradeoff_mem

utils.plot_elasticities_heat(combined, "Elasticity heatmaps", save=True, save_name="PMQuant", sub_folder="Combine/PMQuant")