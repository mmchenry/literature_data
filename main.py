# %% [markdown]
# Interactive coordinate collection
# Specify the path to your image file here
""" Collect coordinates from bitmap images of graphs"""
import collect_coordinates as cc
import give_paths as gp
import os

dir_name = os.path.join('Not yet analyzed','shadwick_Syme_2008', 'fig4')

# Define paths
path = gp.define_paths()

# Define directory containing images
image_path = os.path.join(path['root'], dir_name)

# Collect coordinates for all panels in the directory
cc.all_panels(image_path)


# %% [markdown]
# Interactive coordinate collection
# Specify the path to your image file here
""" Measure body and fin area from bitmap images of fish"""
import measure_areas as ma
import give_paths as gp
import os

dir_name = os.path.join('Not yet analyzed','videler_hess_1984', 'Pollachius_virens')

# Define paths
path = gp.define_paths()

# Define directory containing images
image_path = os.path.join(path['root'], dir_name)

# Number of shapes to measure per image
num_shapes = 2

# Names of the shapes to measure
shape_names = ['body', 'fin']  # Example: ['body', 'fin'] for 2 shapes

# Process all images and measure shapes with bezier curves
ma.all_shapes(image_path, num_shapes=num_shapes, shape_names=shape_names)

# %%
