from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as psub

# Define the path to your data file
path_to_recording_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2')
path_to_rmse_data = path_to_recording_folder/'output_data'/'aligned_data'/'position_rmse_dataframe.csv'
rmse_data = pd.read_csv(path_to_rmse_data)

# Filter out the 'All' marker and focus on specific markers
rmse_data = rmse_data[rmse_data['marker'] != 'All']

# Filter the dataframe for RMSE related to 'x', 'y', and 'z' errors
x_error_df = rmse_data[rmse_data['coordinate'] == 'x_error']
y_error_df = rmse_data[rmse_data['coordinate'] == 'y_error']
z_error_df = rmse_data[rmse_data['coordinate'] == 'z_error']

# Create subplots for each dimension (x, y, z)
fig = psub.make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=('RMSE in X Dimension', 'RMSE in Y Dimension', 'RMSE in Z Dimension'),
    vertical_spacing=0.05  # Reduced vertical spacing
)

# Set the y-axis range for consistency across subplots
y_axis_range = [0, 70]

# X dimension
fig.add_trace(
    go.Bar(x=x_error_df['marker'], y=x_error_df['RMSE'], name='X RMSE', marker_color='royalblue', width=0.6), 
    row=1, col=1
)

# Y dimension
fig.add_trace(
    go.Bar(x=y_error_df['marker'], y=y_error_df['RMSE'], name='Y RMSE', marker_color='seagreen', width=0.6), 
    row=2, col=1
)

# Z dimension
fig.add_trace(
    go.Bar(x=z_error_df['marker'], y=z_error_df['RMSE'], name='Z RMSE', marker_color='tomato', width=0.6), 
    row=3, col=1
)

# Update layout for aesthetics
fig.update_layout(
    height=900, width=1000,
    title_text="RMSE for X, Y, and Z Dimensions for Each Marker",
    title_font=dict(size=20, color='black', family='Arial'),
    showlegend=False,
    plot_bgcolor='white',
    font=dict(size=12),
    margin=dict(l=50, r=50, t=80, b=50)
)

# Set consistent Y-axis range for all subplots and add Y-axis labels
fig.update_yaxes(range=y_axis_range, showgrid=True, gridwidth=0.5, gridcolor='lightgray', title_text="RMSE (mm)")

# Update x-axis to improve readability with bold font
fig.update_xaxes(
    tickangle=45,
    tickfont=dict(size=14, family="Arial", color="black"),  # Bold font for X-axis labels
    showgrid=False
)

# Show the plot
fig.show()
