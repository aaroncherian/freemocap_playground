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

# Find the maximum RMSE value across all dimensions for consistent x-axis range
max_rmse = max(x_error_df['RMSE'].max(), y_error_df['RMSE'].max(), z_error_df['RMSE'].max())

# Create subplots for each dimension (x, y, z) in horizontal layout (3 columns, 1 row)
fig = psub.make_subplots(
    rows=1, cols=3, shared_yaxes=True,  # Horizontal layout
    subplot_titles=('RMSE in X Dimension', 'RMSE in Y Dimension', 'RMSE in Z Dimension'),
    horizontal_spacing=0.05  # Reduced horizontal spacing
)

# X dimension with rounded numbers
fig.add_trace(
    go.Bar(x=x_error_df['RMSE'], y=x_error_df['marker'], name='X RMSE', marker_color='darkred', 
           orientation='h', text=x_error_df['RMSE'].round(1), textposition='auto'),  # Rounded to 1 decimal
    row=1, col=1
)

# Y dimension with rounded numbers
fig.add_trace(
    go.Bar(x=y_error_df['RMSE'], y=y_error_df['marker'], name='Y RMSE', marker_color='seagreen', 
           orientation='h', text=y_error_df['RMSE'].round(1), textposition='auto'),  # Rounded to 1 decimal
    row=1, col=2
)

# Z dimension with darker red and white text
fig.add_trace(
    go.Bar(x=z_error_df['RMSE'], y=z_error_df['marker'], name='Z RMSE', marker_color='royalblue',  # Darker red
           orientation='h', text=z_error_df['RMSE'].round(1), textposition='auto', textfont=dict(color='white')),  # White text
    row=1, col=3
)

# Update layout for aesthetics
fig.update_layout(
    height=800, width=1400,
    title_text="RMSE for X, Y, and Z Dimensions for Each Marker",
    title_font=dict(size=22, color='black', family='Arial'),
    showlegend=False,
    plot_bgcolor='white',
    font=dict(size=16),
    margin=dict(l=50, r=50, t=80, b=50)
)

# Set consistent X-axis range for all subplots and enable vertical gridlines (along the x-axis)
fig.update_xaxes(range=[0, max_rmse], title_text="RMSE (mm)", showgrid=True, gridwidth=0.5, gridcolor='lightgray')

# Set Y-axis only for the first plot (X dimension), increase the font size for marker names
fig.update_yaxes(showgrid=False, title_text="Markers", row=1, col=1)
fig.update_yaxes(showticklabels=False, row=1, col=2)  # Remove Y labels for Y dimension
fig.update_yaxes(showticklabels=False, row=1, col=3)  # Remove Y labels for Z dimension

# Update Y-axis tick font size for bigger marker names
fig.update_yaxes(tickfont=dict(size=14))

# Show the plot
fig.show()
