import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

freemocap_path_lengths = pd.DataFrame({
  "Eyes Open/Solid Ground": 0.17767209079046525,
  "Eyes Closed/Solid Ground": 0.2290439249113198,
  "Eyes Open/Foam": 0.23463957463731774,
  "Eyes Closed/Foam": 0.27424361761227967,
  "System": 'freemocap'
 }, index = [0])

qualisys_path_lengths = pd.DataFrame({
  "Eyes Open/Solid Ground": 0.11225410557470969,
  "Eyes Closed/Solid Ground": 0.17094917664131196,
  "Eyes Open/Foam": 0.1753038667711608,
  "Eyes Closed/Foam": 0.22312323644775237,
  "System": 'qualisys'
}, index = [0])


df_merged = pd.concat([freemocap_path_lengths,qualisys_path_lengths], ignore_index=False, sort = False)
df_melted = pd.melt(df_merged, id_vars = 'System', value_vars = ['Eyes Open/Solid Ground', 'Eyes Closed/Solid Ground', 'Eyes Open/Foam', 'Eyes Closed/Foam'], var_name = 'Condition', value_name = 'COM_Path_Length')

sns.set_theme(style="whitegrid")

# ax = sns.barplot(
#     data=df_melted,
#     x="Condition", y="COM_Path_Length", hue="System",
#     errorbar="sd", palette="dark", alpha = .6,
# )


# for i in ax.containers:
#     ax.bar_label(i,


ax = sns.lineplot(
    data=df_melted,
    x="Condition", y="COM_Path_Length", hue="System",
    errorbar="sd", palette="dark", alpha=.6
)

sns.despine(left=True)
ax.set(ylabel = 'Center of Mass Path Length (mm)')
ax.legend(title = '')


plt.show()
f = 2


