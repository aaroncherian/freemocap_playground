from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Define the path to your data file
path_to_recording_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2')
path_to_rmse_data = path_to_recording_folder/'output_data'/'aligned_data'/'position_rmse_dataframe.csv'
rmse_data = pd.read_csv(path_to_rmse_data)

# Filter out the 'All' marker and focus on specific markers
rmse_data = rmse_data[rmse_data['dimension'] == 'Per Dimension']

# Extract total RMSE for each dimension
total_rmse_x = rmse_data[rmse_data['coordinate'] == 'x_error']['RMSE'].round(1).values[0]
total_rmse_y = rmse_data[rmse_data['coordinate'] == 'y_error']['RMSE'].round(1).values[0]
total_rmse_z = rmse_data[rmse_data['coordinate'] == 'z_error']['RMSE'].round(1).values[0]

# Create a DataFrame for X, Y, and Z total RMSE
total_rmse_df = pd.DataFrame({
    'Dimension': ['X', 'Y', 'Z'],
    'Total RMSE': [total_rmse_x, total_rmse_y, total_rmse_z]
})

# Create a bar plot for the total RMSE in X, Y, and Z dimensions
fig = go.Figure()

fig.add_trace(go.Bar(
    x=total_rmse_df['Dimension'], 
    y=total_rmse_df['Total RMSE'], 
    marker_color=['darkred', 'seagreen', 'royalblue'],  # Colors for X, Y, Z
    text=total_rmse_df['Total RMSE'],  # Show RMSE values
    textposition='auto'
))

# Update layout for aesthetics, including larger labels and horizontal gridlines
fig.update_layout(
    title="Total RMSE for X, Y, and Z Dimensions",
    xaxis_title="Dimension",
    yaxis_title="Total RMSE (mm)",
    title_font=dict(size=20, color='black', family='Arial'),
    plot_bgcolor='white',
    font=dict(size=26),
    height=1000, width=1000,
    margin=dict(l=50, r=50, t=80, b=50),
    xaxis=dict(
        tickfont=dict(size=30)  # Larger font for X, Y, Z labels
    ),
    yaxis=dict(
        showgrid=True, gridwidth=0.5, gridcolor='lightgray'  # Horizontal gridlines
    )
)

# Show the plot
fig.show()
