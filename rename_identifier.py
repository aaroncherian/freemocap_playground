from pathlib import Path

def rename_folder_and_files(folder_path, old_id, new_id):
    try:
        # Rename folder
        folder_name = folder_path.name
        if old_id in folder_name:
            new_folder_name = folder_name.replace(old_id, new_id)
            new_folder_path = folder_path.parent / new_folder_name
            if new_folder_path.exists():
                print(f"Warning: Target folder {new_folder_path} already exists. Skipping folder rename.")
            else:
                folder_path.rename(new_folder_path)
                print(f"Renamed folder {folder_name} to {new_folder_name}")
                folder_path = new_folder_path  # Update the folder path
        
        # Rename files and folders inside the folder recursively
        for path in folder_path.rglob('*'):
            name = path.name
            if old_id in name:
                new_name = name.replace(old_id, new_id)
                new_path = path.parent / new_name
                if new_path.exists():
                    print(f"Warning: Target path {new_path} already exists. Skipping rename.")
                else:
                    path.rename(new_path)
                    print(f"Renamed {name} to {new_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Paths to directories
path_to_freemocap_recordings = Path(r'D:\2023-06-07_JH')

# List folders in directory
freemocap_folders = [f for f in path_to_freemocap_recordings.iterdir() if f.is_dir()]

# Iterate through folders and rename them along with their files
for folder in freemocap_folders:
    rename_folder_and_files(folder, 'JH', 'TF01')
