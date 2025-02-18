import numpy as np
import pandas as pd

def csv_to_trc(csv_file, output_trc_path, sampling_freq):
    """
    Converts a CSV file containing marker data into a TRC file for OpenSim.

    Parameters:
        csv_file (str): Path to the CSV file.
        output_trc_path (str): Path to save the TRC file.
        sampling_freq (float): Sampling frequency of the motion capture data.
    """
    # Load CSV
    df = pd.read_csv(csv_file)

    # Extract time column
    if 'unix_timestamps' in df.columns:
        time_column = df['unix_timestamps'] - df['unix_timestamps'].iloc[0]  # Normalize time
    elif 'Time' in df.columns:
        time_column = df['Time']
    else:
        raise ValueError("CSV must contain either 'Time' or 'unix_timestamps'.")

    # Extract marker data (columns that are not Time, Frame, or unix_timestamps)
    excluded_columns = {'Frame', 'Time', 'unix_timestamps'}
    marker_columns = [col for col in df.columns if col not in excluded_columns]

    # Extract unique marker names (without axis labels)
    marker_names = sorted(set(col.rsplit(' ', 1)[0] for col in marker_columns))

    # Ensure markers are formatted correctly in XYZ triplets
    ordered_columns = [f"{m} {axis}" for m in marker_names for axis in ['X', 'Y', 'Z']]
    if not all(col in df.columns for col in ordered_columns):
        missing_cols = [col for col in ordered_columns if col not in df.columns]
        raise ValueError(f"Marker data is missing expected columns: {missing_cols}")

    # Extract marker positions
    marker_data = df[ordered_columns].to_numpy()

    # Create the TRC DataFrame
    trc_df = pd.DataFrame(data=np.column_stack([time_column, marker_data]), columns=['Time'] + ordered_columns)

    # Save TRC file
    save_trc(trc_df, marker_names, output_trc_path, sampling_freq)


def save_trc(df, labels, output_path, sampling_freq):
    """
    Saves the TRC DataFrame to a properly formatted TRC file for OpenSim.

    Parameters:
        df (pd.DataFrame): The TRC data as a DataFrame.
        labels (list): List of marker names.
        output_path (str): Path to save the TRC file.
        sampling_freq (float): Sampling frequency of the data.
    """
    n_markers = len(labels)

    # TRC Header
    header_lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_path}\n",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
        f"{sampling_freq:.2f}\t{sampling_freq:.2f}\t{df.shape[0]}\t{n_markers}\tmm\t{sampling_freq:.2f}\t1\t{df.shape[0]}\n",
        "Frame#\tTime\t" + "\t\t\t".join(labels) + "\n",
        "\t\t" + "\t".join(["X\tY\tZ"] * n_markers) + "\n"
    ]

    # Write header and data to TRC file
    with open(output_path, 'w') as trc_file:
        trc_file.writelines(header_lines)
        df.to_csv(trc_file, sep='\t', index=True, header=False, float_format="%.6f")

    print(f"TRC file saved to {output_path}")


csv_file_path = r"D:\mdn_data\sesh_2023-05-17_13_48_44_MDN_treadmill_2\output_data\component_qualisys_original\marker_data_synced.csv"
output_trc_path = r"D:\mdn_data\sesh_2023-05-17_13_48_44_MDN_treadmill_2\output_data\component_qualisys_original\marker_data_synced.trc"
sampling_freq = 300  # Hz
csv_to_trc(csv_file_path, output_trc_path, sampling_freq)
