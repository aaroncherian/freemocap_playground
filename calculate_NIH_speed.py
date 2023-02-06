
from freemocap_utils.GUI_widgets.NIH_widgets.path_length_tools import PathLengthCalculator
from freemocap_utils import freemocap_data_loader
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.collections import PathCollection

def calculate_magnitude(point):
    speed = np.sqrt((point[0])**2 + (point[1])**2 + (point[2])**2)
    return speed

session_path = Path(r'D:\ValidationStudy_aaron\FreeMoCap_Data\sesh_2022-11-02_13_55_55_atc_nih_balance')

#session_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data\sesh_2022-05-24_16_02_53_JSM_T1_NIH')

data_holder = freemocap_data_loader.FreeMoCapDataLoader(session_path)


COM_data = data_holder.load_total_body_COM_data()





velocity_COM = np.diff(COM_data, axis = 0)

speed_list = []

for x in range(velocity_COM.shape[0]):
    speed = calculate_magnitude(velocity_COM[x])
    speed_list.append(speed)


eo_sg_range = [600,2170]
ec_sg_range = [2950,4450]

eo_fg_range = [5050,6550]
ec_fg_range = [7000, 8500]


##for jon's data
# eo_sg_range = [700,2350]
# ec_sg_range = [2750,4400]

# eo_fg_range = [6000,7750]
# ec_fg_range = [8500, 9850]

eo_sg_data = speed_list[eo_sg_range[0]: eo_sg_range[1]]
ec_sg_data = speed_list[ec_sg_range[0]: ec_sg_range[1]]
eo_fg_data = speed_list[eo_fg_range[0]: eo_fg_range[1]]
ec_fg_data = speed_list[ec_fg_range[0]: ec_fg_range[1]]

condition_dict = {'EO/SG':eo_sg_data, 'EC/SG':ec_sg_data, 'EO/FG':eo_fg_data, 'EC/FG':ec_fg_data}





condition_df = pd.DataFrame({ key:pd.Series(value) for key, value in condition_dict.items()})


ax = sns.violinplot(data = condition_df)

ax = sns.violinplot(data = condition_df, cut = 1, color = ".8")

for artist in ax.lines:
    artist.set_zorder(10)
for artist in ax.findobj(PathCollection):
    artist.set_zorder(11)

ax = sns.stripplot(data = condition_df, jitter = True, zorder=1, alpha = .3)

ax.set_xlabel('Condition')
ax.set_ylabel('COM Speed')

ax.set_title('Speed Plot')

fig = ax.get_figure()


plt.show()


f = 2


