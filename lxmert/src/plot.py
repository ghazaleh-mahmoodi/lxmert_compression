import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import dataframe_image as dfi
from matplotlib.patches import Patch
sns.set()


df = pd.read_json('result/lxmert_experiment.json')
columns_list = sorted(df.columns.to_list())

sebnet = [s for s in columns_list if 'Low_Magnitude' in s]

re=[]
for sparcity_level in np.arange(10, 100, 10):
    result = []
    result.append(sparcity_level)
    for subnet_mode in ['Low_Magnitude', 'Random', 'High_Magnitude']:
        sebnet = [s for s in columns_list if f'{subnet_mode}-{sparcity_level}' in s][0]
        sebnet = df[sebnet].to_dict()
        result.append(str(sebnet['pruning_result'][list(sebnet['pruning_result'].keys())[-2]]+0.1))
        result.append(sebnet['retrain_result']['Epoch 3 Valid'])
    # print(len(result))
    print(result)
    re.append(result)
# re.append(df['vqa_lxmert_finetuned_seed0/finetune_result.json'].to_dict()['Epoch 3 Valid'])

# print(re)
# sebnet = pd.DataFrame(re)
sebnet = pd.DataFrame(re, columns=['Sparcity Level','Low Magnitude(pruned)','Low Magnitude (retrain)','Random (pruned)','Random (retrain)','High_Magnitude (pruned)','High Magnitude (retrain)'])
# sebnet = sebnet.reset_index(drop=True, inplace=True)
print(sebnet.head())
dfi.export(sebnet, '../report/images/report.PNG')

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

experiment_names = ['Low Magnitude Pruning Subnet (retrained)', 'Low Magnitude Pruning Subnet (pruned)',
                     'Random Pruning Subnet (retrained)', 'Random Pruning Subnet (retrained)', 
                     'High Magnitude Pruning Subnet (retrained)', 'High Magnitude Pruning Subnet (retrained)',
                     'Unpruned Baseline']

def find_result(sparcity):
    result = []
    df = pd.read_json('result/lxmert_experiment.json')
    columns_list = sorted(df.columns.to_list())
    for subnet_mode in ['Low_Magnitude', 'Random', 'High_Magnitude']:
        sebnet = [s for s in columns_list if f'{subnet_mode}-{sparcity_level}' in s][0]
        sebnet = df[sebnet].to_dict()
        result.append(sebnet['retrain_result']['Epoch 3 Valid'])
        result.append(sebnet['pruning_result'][list(sebnet['pruning_result'].keys())[-2]]+0.1)
    result.append(df['vqa_lxmert_finetuned_seed0/finetune_result.json'].to_dict()['Epoch 3 Valid'])
    
    print(result)
    return result

fig, axs = plt.subplots(3, 3, figsize=(9,9))
my_cmap = sns.color_palette("Paired") + sns.color_palette("Paired")[:8]

patterns = [ r"*" ,  r"|", r"\\" , r"\\||" , r"--", r"--||", r"//", r"//||", "xx", "xx||", "..", "..||", "oo", "oo||"]
# x_pos = np.arange(1, 8, 1)
x_pos = np.arange(len(experiment_names))
bbox_to_anchor_left=0.6

for i, sparcity_level in enumerate(np.arange(10, 100, 10)):
    means = find_result(sparcity_level)
    print(len(means), len(x_pos))

    row = i // 3
    col = i % 3
    print(row, col)
    
    axs[row, col].bar(x_pos, means, 0.7,  align='center', color=my_cmap)#my_cmap(my_norm(range(len(x_pos)))))   
    axs[row, col].set_ylabel('Val  Accuracy')
    
    axs[row, col].set_title(f'VQA ({sparcity_level}%)')
    axs[row, col].set_xticks([])
    
    bars = axs[row, col].patches
    
    for bar, hatch in zip(bars, patterns):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)

legend_elements = [Patch(facecolor=my_cmap[i], hatch=patterns[i], label=exp) for i, exp in enumerate(experiment_names)]
l_col = 3
legend = plt.legend(flip(legend_elements, l_col), flip(experiment_names, l_col), loc='best', ncol=l_col, bbox_to_anchor=(0.93, -0.1), labelspacing=1.5, handlelength=4)

for patch in legend.get_patches():
    patch.set_height(10)
    patch.set_y(-1)

plt.subplots_adjust(right=1.5)
fig.tight_layout()
# plt.show()
fig.savefig('../report/images/experiment_result.PNG', bbox_inches='tight')


#---------------------------------------------------------------------------------------------------------------------------#    


for subnet_mode in ['High_Magnitude', 'Low_Magnitude', 'Random']:
    subnetwork = [s for s in columns_list if subnet_mode in s and 'seed1' in s]
    print("subnetwork : ", subnetwork)

    retrain_result = []
    reset_initial_weight = []
    pruned = []
    for sub_member in subnetwork:
        member = df[sub_member].to_dict()
        retrain_result.append(member['retrain_result']['Epoch 3 Valid'])
        reset_initial_weight.append(member['pruning_result']['accuarcy after pruning'])
        pruned.append(member['pruning_result'][list(member['pruning_result'].keys())[-2]])

    sparcity_level = np.arange(10, 100, 10)
    baseline_model = [df['vqa_lxmert_finetuned_seed0/finetune_result.json'].to_dict()['Epoch 3 Valid']]*9

    with plt.style.context('ggplot'):
        #['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 4, 5, 6, 7, 8, 9, 10, 11]
        # fig = plt.figure(figsize=(7,10))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel('Val  Accuracy')
        ax.set_xlabel('Sparcity Level (%)')
        name_plot = subnet_mode.replace('_', ' ')
        ax.set_title(f'{name_plot} Pruning LXMERT On VQA')

        # ax.plot(sparcity_level, retrain_result, marker = "<", label = 'Low Magnitude Subnetwork (retrained)')
        # ax.plot(sparcity_level, reset_initial_weight, marker = "h", label = 'Low Magnitude Subnetwork (reset initial weight)')
        # ax.plot(sparcity_level, pruned, marker = "*", label = 'Low Magnitude Subnetwork (pruned)')
        # ax.plot(sparcity_level, baseline_model,marker = ".", label = 'Unpruned Baseline')

        ax.plot(sparcity_level, retrain_result, marker = ">", label = 'retrained')
        ax.plot(sparcity_level, reset_initial_weight, marker = "h", label = 'reset initial weight')
        var = -0.05
        if subnet_mode != 'High_Magnitude':
            ax.plot(sparcity_level, pruned, marker = "*", label = 'pruned')
            var = 0
        ax.plot(sparcity_level, baseline_model, '--r', label = 'Unpruned Baseline')

        plt.legend(loc = "best", ncol=4, bbox_to_anchor=(0.95+var, -0.2), labelspacing=1, handlelength=4)
        plt.subplots_adjust(right=1.5)
        plt.savefig(f'../report/images/{subnet_mode}_experiment_result.PNG', bbox_inches='tight')
        plt.cla()
        plt.clf()
        # plt.show()

# dfi.export(df, '../reports/tokenization.png')
