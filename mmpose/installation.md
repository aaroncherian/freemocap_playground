uv pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# torch needs to be pinned to 2.1 to be compatible with mmcv (at least I know 2.5 is not compatible)

uv pip uninstall numpy
uv pip install numpy==1.26.4

uv pip install mmengine
mim install mmengine

uv pip install git+https://github.com/open-mmlab/mmpose.git --no-deps

# not sure if these are needed, just recommended by the bot
uv pip install opencv-contrib-python pillow scipy tqdm matplotlib
uv pip install xtcocotools json_tricks munkres
