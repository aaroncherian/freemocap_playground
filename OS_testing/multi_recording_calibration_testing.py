from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from calibration_translation_comparison import run_pairwise_distance_calculation
from calibration_rotation_comparison import run_pairwise_rotation_calculation

# Define the calibration folder path
path_to_multi_calibration_folder = Path(r'D:\system_testing\calibrations')
system_colors = {"macos": "green", "ubuntu": "darkorange", "windows": "blue"}

# Process distance and rotation data for all recordings
data_list = []
for count, folder in enumerate(path_to_multi_calibration_folder.iterdir()):
    # Distance calculations
    df_distance = run_pairwise_distance_calculation(path_to_folder_of_tomls=folder, num_cams=3)
    df_distance['id'] = f'recording_{count}'
    df_distance['metric'] = 'distance'
    df_distance['ranks'] = [1,2,3]
    data_list.append(df_distance)

    # Rotation calculations
    df_rotation = run_pairwise_rotation_calculation(path_to_folder_of_tomls=folder, num_cams=3)
    df_rotation['id'] = f'recording_{count}'
    df_rotation['metric'] = 'rotation'
    df_rotation['ranks'] = [1,2,3]
    data_list.append(df_rotation)

# Combine all data into a single DataFrame
df_combined = pd.concat(data_list, ignore_index=True)

# Reshape the DataFrame so that each system is a row with corresponding metrics
df_melted = df_combined.melt(id_vars=['ranks', 'id', 'metric'], 
                             value_vars=['macos', 'ubuntu', 'windows'], 
                             var_name='system', 
                             value_name='value')

# Compute mean and standard deviation **across all recordings** for each system per rank
grouped_stats = df_melted.groupby(['ranks', 'system', 'metric'])['value'].agg(['mean', 'std']).reset_index()
grouped_stats.columns = ['ranks', 'system', 'metric', 'mean', 'std']

# Generate and display the distance summary table
distance_table = grouped_stats[grouped_stats['metric'] == 'distance'].pivot(index='system', columns='ranks', values='mean')

# Rename columns for clarity
distance_table.columns = [f'Rank {col}' for col in distance_table.columns]

# Display the table
print(distance_table)

# Create subplot layout with 2 rows (top: distance, bottom: rotation)
fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       subplot_titles=("Mean Distance per Rank (Across All Recordings)", 
                                       "Mean Rotation per Rank (Across All Recordings)"))

# Loop over each system and plot bars for distance & rotation
for system in grouped_stats['system'].unique():
    color = system_colors.get(system, "gray")  # Default to gray if system not in dictionary

    system_data_dist = grouped_stats[(grouped_stats['system'] == system) & (grouped_stats['metric'] == 'distance')]
    system_data_rot = grouped_stats[(grouped_stats['system'] == system) & (grouped_stats['metric'] == 'rotation')]

    # Distance plot (Row 1)
    fig.add_trace(go.Bar(
        x=system_data_dist['ranks'],
        y=system_data_dist['mean'],
        name=f"{system} - Distance",
        marker_color=color,  # Set color
        error_y=dict(type='data', array=system_data_dist['std'], visible=True),
        text=[f"{val:.2f}" for val in system_data_dist['mean']],  # Display mean values
        textposition='outside'
    ), row=1, col=1)

    # Rotation plot (Row 2)
    fig.add_trace(go.Bar(
        x=system_data_rot['ranks'],
        y=system_data_rot['mean'],
        name=f"{system} - Rotation",
        marker_color=color,  # Set color
        error_y=dict(type='data', array=system_data_rot['std'], visible=True),
        text=[f"{val:.2f}" for val in system_data_rot['mean']],  # Display mean values
        textposition='outside'
    ), row=2, col=1)

# Update layout for clarity
fig.update_layout(
    title="System vs. System Comparison (Mean Distance & Rotation per Rank, Across All Recordings)",
    xaxis_title="Rank",
    yaxis_title="Mean Distance",
    xaxis2_title="Rank",
    yaxis2_title="Mean Rotation",
    barmode='group',
    legend_title="System",
    showlegend=True
)

# Show plot
fig.show()
