import sys
import os

def define_paths():

    # Initialize the path dictionary
    path = {}

    # Path on Matt's mac
    if sys.platform == "darwin" and os.path.isdir("/Users/mmchenry/Documents"):  # Check if running on macOS

        path['root'] = "/Users/mmchenry/Documents/Projects/fish_swimming_literature_data"

    else:
        raise ValueError(f"Do not recognize this system.")


    return path