import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
import numpy as np
import seaborn as sns

def aggregate_path_lengths(trials):
  
    trials_dataframe = pd.DataFrame(trials)

    trials_mean = trials_dataframe.mean()
    trials_std = trials_dataframe.std()

    return trials_mean, trials_std, trials_dataframe

# Example usage


trial_1 = {'Condition 2/Condition 1':.71, 'Condition 4/Condition 1':2.34}
trial_2 = {'Condition 2/Condition 1':1.12, 'Condition 4/Condition 1':3.79}
trial_3 = {'Condition 2/Condition 1':1.26, 'Condition 4/Condition 1':3.16}

trial_data = [trial_1, trial_2, trial_3]

trials_mean, trials_std, trials_dataframe = aggregate_path_lengths(trial_data)

# Conditions

conditions = trials_dataframe.columns

# Create figure with subplots
fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

# Plotting Freemocap data
for index, row in trials_dataframe.iterrows():
    ax.plot(conditions, row, '-o', color='#72117c', alpha=0.6)

# Adding Freemocap mean and error bars
ax.errorbar(conditions, trials_mean, yerr=trials_std, fmt='-o', color='black', capsize=5, label='Mean')

# Labels and titles for Freemocap
# ax.set_title('Freemocap')
ax.set_ylabel('Ratio Score', fontsize=12)
ax.set_xlabel('Condition Ratio', fontsize=12)
ax.legend(loc = 'upper left')
plt.xticks(fontsize=12)

# axs[0].set_ylim([0, .6])


fig.suptitle('NIH Toolbox Ratio Scores', fontsize=16)

# Display the plot
plt.tight_layout()
plt.show()