import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# 1) Function to create single-marker figure with X/Y/Z in legend
def make_marker_figure(data_dict, marker_idx, marker_name, dash_styles=['solid','dash','dot']):
    pipelines = list(data_dict.keys())
    axis_rgb = {'X': (255,0,0), 'Y': (0,128,0), 'Z': (0,0,255)}
    factors = np.linspace(1, 0.5, len(pipelines))
    pipeline_colors = {
        p: {axis: f'rgba({int(r*f)},{int(g*f)},{int(b*f)},0.7)' 
            for axis, (r,g,b) in axis_rgb.items()}
        for p,f in zip(pipelines, factors)
    }
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        subplot_titles=[f"{marker_name}: X", f"{marker_name}: Y", f"{marker_name}: Z"])
    n_frames = next(iter(data_dict.values())).shape[0]
    time = np.arange(n_frames)
    # Add traces with axis-specific legend entries
    for dim, axis in enumerate(['X','Y','Z']):
        for pi, p in enumerate(pipelines):
            fig.add_trace(go.Scatter(
                x=time,
                y=data_dict[p][:, marker_idx, dim],
                mode='lines',
                name=f"{p} {axis}",
                showlegend=True,
                legendgroup=f"{p}_{axis}",
                line=dict(
                    color=pipeline_colors[p][axis],
                    dash=dash_styles[pi % len(dash_styles)],
                    width=2
                )
            ), row=dim+1, col=1)
        # tight y-axis
        vals = np.concatenate([data_dict[p][:,marker_idx,dim] for p in pipelines])
        mn, mx = np.nanmin(vals), np.nanmax(vals)
        pad = 0.05*(mx-mn)
        fig.update_yaxes(range=[mn-pad, mx+pad], row=dim+1, col=1)
        if dim == 2:
            fig.update_xaxes(title_text='Frame', row=3, col=1)
        else:
            fig.update_xaxes(showticklabels=False, row=dim+1, col=1)

    fig.update_layout(template='plotly_white', height=600, width=1800,
                      margin=dict(t=50, r=150))
    return fig

if __name__ == "__main__":
    # 2) Load dataset (preserved paths)
    paths = {
        'thresholded_filtered': Path(r"D:\ferret_em_talk\ferret_04_28_ears_nose\output_data\dlc_body_rigid_3d_xyz_thresholded_filtered.npy"),
        'thresholded_no_filter': Path(r"D:\ferret_em_talk\ferret_04_28_ears_nose\output_data\dlc_body_rigid_3d_xyz_thresholded_no_filter.npy"),
        'no_threshold_no_filter': Path(r"D:\ferret_em_talk\ferret_04_28_ears_nose\output_data\dlc_body_rigid_3d_xyz_no_threshold_no_filter.npy")
    }
    data_dict = {k: np.load(v) for k,v in paths.items()}
    markers = ['nose','right_ear','left_ear','toy']

    # 3) Generate individual figures
    figs = [make_marker_figure(data_dict, idx, name) for idx, name in enumerate(markers)]

    # 4) Build combined HTML
    html_snippets = []
    html_snippets.append(figs[0].to_html(full_html=False, include_plotlyjs='cdn'))
    for fig in figs[1:]:
        html_snippets.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # Wrap into HTML page
    html_page = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>All Marker Trajectories</title>
    </head>
    <body>
    <h1>All Marker Trajectories</h1>
    {' '.join(html_snippets)}
    </body>
    </html>
    """

    # Write to file
    with open("all_markers_combined.html", "w") as f:
        f.write(html_page)

    print("✅ Combined HTML saved as all_markers_combined.html")

# if __name__ == "__main__":

#     from pathlib import Path
#     data_threshold_filter_path     = Path(r"D:\ferret_em_talk\ferret_04_28_ears_nose\output_data\dlc_body_rigid_3d_xyz_thresholded_filtered.npy")
#     data_threshold_no_filter_path  = Path(r"D:\ferret_em_talk\ferret_04_28_ears_nose\output_data\dlc_body_rigid_3d_xyz_thresholded_no_filter.npy")
#     data_no_threshold_no_filter_path = Path(r"D:\ferret_em_talk\ferret_04_28_ears_nose\output_data\dlc_body_rigid_3d_xyz_no_threshold_no_filter.npy")

#     data_dict = {
#         "thresholded_filtered":    np.load(data_threshold_filter_path),
#         "thresholded_no_filter":   np.load(data_threshold_no_filter_path),
#         "no_threshold_no_filter":  np.load(data_no_threshold_no_filter_path),
#     }

#     markers = ["nose", "right_ear", "left_ear", "toy"]
#     n_frames = next(iter(data_dict.values())).shape[0]
#     time = np.arange(n_frames)

#     # plot for each marker, all on the same graph per marker
#     plot_all_markers_vertical(data_dict, markers)

    