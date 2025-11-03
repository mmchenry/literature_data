# %% [markdown]
# Interactive coordinate collection
# Specify the path to your image file here
""" Collect """
import collect_coordinates as cc
import give_paths as gp
import os

dir_name = os.path.join('bainbridge_1958', 'fig10')

# Define paths
path = gp.define_paths()

# Define directory containing images
image_path = os.path.join(path['root'], 'captured_images', dir_name)

# Collect coordinates for all panels in the directory
cc.all_panels(image_path)


# %%
