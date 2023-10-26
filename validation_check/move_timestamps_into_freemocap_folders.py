from pathlib import Path
import shutil

# Paths to directories
path_to_pre_alpha_recordings = Path(r'D:\2023-05-17_MDN_NIH_data\FreeMocap_Data')
path_to_freemocap_recordings = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3')

# List folders in each directory
pre_alpha_folders = {f.name for f in path_to_pre_alpha_recordings.iterdir() if f.is_dir()}
freemocap_folders = {f.name for f in path_to_freemocap_recordings.iterdir() if f.is_dir()}

# Find common folders
common_folders = pre_alpha_folders.intersection(freemocap_folders)

# Iterate through common folders
for folder in common_folders:
    source_file = path_to_pre_alpha_recordings / folder / "unix_synced_timestamps.csv"
    target_file = path_to_freemocap_recordings / folder / 'synchronized_videos'/ 'timestamps'/ "unix_synced_timestamps.csv"

    #these next two lines are if you need to create extra folders to save in
    target_directory = target_file.parent  # get the parent directory of the target file 
    # Create the target directory if it does not exist
    target_directory.mkdir(parents=True, exist_ok=True)
    
    # Check if the source file exists before copying
    if source_file.exists():
        shutil.copy(str(source_file), str(target_file))
        print(f"Copied {source_file} to {target_file}")
    else:
        print(f"Source file {source_file} does not exist.")
