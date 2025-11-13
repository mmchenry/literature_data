import sys
import os

def define_paths():

    # Initialize the path dictionary
    path = {}

    # Path on Matt's mac
    if sys.platform == "darwin" and os.path.isdir("/Users/mmchenry/Google Drive"):  # Check if running on macOS

        path['root'] = "/Users/mmchenry/Google Drive/Shared drives/Flow vis/Videos & data for Mittal/Kinematics from the literature"

    else:
        raise ValueError(f"Do not recognize this system.")


    return path