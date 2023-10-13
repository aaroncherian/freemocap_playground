from pathlib import Path
import shutil

# Paths to directories
path_to_pre_alpha_recordings = Path('D:/2023-06-07_JH/FreeMocap_Data')
path_to_freemocap_recordings = Path('D:/2023-06-07_JH/1.0_recordings/treadmill_calib')

# List folders in each directory
pre_alpha_folders = {f.name for f in path_to_pre_alpha_recordings.iterdir() if f.is_dir()}
freemocap_folders = {f.name for f in path_to_freemocap_recordings.iterdir() if f.is_dir()}

# Find common folders
common_folders = pre_alpha_folders.intersection(freemocap_folders)

# Iterate through common folders
for folder in common_folders:
    source_file = path_to_pre_alpha_recordings / folder / f"{folder}_framerate.txt"
    target_file = path_to_freemocap_recordings / folder / f"{folder}_framerate.txt"
    
    # Check if the source file exists before copying
    if source_file.exists():
        shutil.copy(str(source_file), str(target_file))
        print(f"Copied {source_file} to {target_file}")
    else:
        print(f"Source file {source_file} does not exist.")
