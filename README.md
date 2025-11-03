# Literature Data - Interactive Coordinate Collection

This project provides tools for interactively selecting coordinates from images with axis calibration, useful for data collection tasks in literature analysis and research.

## Features

- **Interactive point selection**: Click on images to select coordinates with visual feedback
- **Visual markers**: Selected points are marked with red circles and numbered labels
- **Axis calibration**: Calibrate x and y axes to convert pixel coordinates to real-world values
- **Batch processing**: Process multiple images in a directory automatically
- **CSV export**: All calibration constants and coordinates saved to a single CSV file
- **Point removal**: Right-click to remove the last selected point
- **Comprehensive data storage**: Stores calibration constants, units, body length, origin pixels, and coordinates

## Setup

### Prerequisites

- [Mamba](https://mamba.readthedocs.io/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed

### Installation

1. Create the conda environment using the provided `environment.yml`:
   ```bash
   mamba env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   mamba activate lit_data
   ```

3. Verify installation:
   ```bash
   python -c "import matplotlib, numpy, pandas; print('All packages installed successfully!')"
   ```

## Usage

### Batch Processing Multiple Images (Recommended)

The `all_panels()` function processes all images in a directory, calibrates axes, and collects coordinates for each image.

1. Open `main.py` and set the directory path containing your images:
   ```python
   import collect_coordinates as cc
   import give_paths as gp
   import os
   
   path = gp.define_paths()
   image_dir = os.path.join(path['root'], 'captured_images', 'your_folder')
   
   # Process all images in the directory
   cc.all_panels(image_dir)
   ```

2. Run the script:
   ```bash
   python main.py
   ```

3. For each image, you'll be prompted to:
   - **Calibrate X-axis**: 
     - Click on the START point of the x-axis
     - Click on the END point of the x-axis
     - Press Enter to confirm
     - Enter the x-axis value at the START point
     - Enter the x-axis value at the END point
     - Enter the units (e.g., 'cm', 'mm', 'm')
   
   - **Calibrate Y-axis**: 
     - Click on the START point of the y-axis
     - Click on the END point of the y-axis
     - Press Enter to confirm
     - Enter the y-axis value at the START point
     - Enter the y-axis value at the END point
     - Enter the units (e.g., 'cm', 'mm', 'm')
   
   - **Enter body length**: 
     - Enter the body length in centimeters for the image
   
   - **Collect coordinates**:
     - Left click on the image to select points (red circles appear)
     - Right click to remove the last selected point
     - Press Enter or close the window when done

4. All data is saved to `all_panels_coordinates.csv` in the image directory.

### Single Image Processing

For processing a single image with calibration:

```python
import collect_coordinates as cc

image_path = "path/to/your/image.png"

# Calibrate both axes
x_calibration = cc.calibrate_axis(image_path, axis='x')
y_calibration = cc.calibrate_axis(image_path, axis='y')

# Collect coordinates with calibration
coordinates_df = cc.collect_coordinates(image_path, 
                                       x_calibration=x_calibration,
                                       y_calibration=y_calibration)
```

### Single Image Without Calibration

For simple coordinate collection without calibration:

```python
import collect_coordinates as cc

image_path = "path/to/your/image.png"

# Collect coordinates (no calibration)
coordinates_df = cc.collect_coordinates(image_path)
```

Coordinates will be saved as `{image_name}_coordinates.csv` in the same directory as the image.

## Interactive Controls

### During Calibration

- **Left click**: Select a point (START or END of axis)
- **Press Enter**: Confirm calibration and proceed to value entry
- **Press 'q'**: Cancel calibration

### During Coordinate Collection

- **Left click**: Select a coordinate point (red circle appears with number)
- **Right click**: Remove the last selected point
- **Press Enter**: Finish and save coordinates
- **Close window**: Also finishes and saves coordinates

## CSV Output Structure

When using `all_panels()`, the output CSV (`all_panels_coordinates.csv`) contains the following columns:

### Image Information
- `image_filename`: Name of the source image file

### Body Length
- `body_length_cm`: Body length in centimeters for the image

### X-Axis Calibration
- `x_cal_start_pixel`: X-coordinate of start point in pixels
- `x_cal_end_pixel`: X-coordinate of end point in pixels
- `x_cal_start_value`: Real-world value at start point
- `x_cal_end_value`: Real-world value at end point
- `x_cal_units`: Units for x-axis (e.g., 'cm', 'mm', 'm')
- `x_origin_pixel_x`: X-coordinate of x-axis origin in pixels
- `x_origin_pixel_y`: Y-coordinate of x-axis origin in pixels

### Y-Axis Calibration
- `y_cal_start_pixel`: Y-coordinate of start point in pixels
- `y_cal_end_pixel`: Y-coordinate of end point in pixels
- `y_cal_start_value`: Real-world value at start point
- `y_cal_end_value`: Real-world value at end point
- `y_cal_units`: Units for y-axis (e.g., 'cm', 'mm', 'm')
- `y_origin_pixel_x`: X-coordinate of y-axis origin in pixels
- `y_origin_pixel_y`: Y-coordinate of y-axis origin in pixels

### Coordinates
- `x`: Calibrated x-coordinate value
- `y`: Calibrated y-coordinate value
- `x_pixel`: Original x-coordinate in pixels
- `y_pixel`: Original y-coordinate in pixels

## Function Documentation

### `all_panels(image_dir_path)`

Process all image files in a directory with calibration and coordinate collection.

**Parameters:**
- `image_dir_path` (str): Path to directory containing image files

**Returns:**
- `pandas.DataFrame`: Combined DataFrame with all calibration constants and coordinates

**Output:**
- CSV file: `all_panels_coordinates.csv` in the image directory

### `calibrate_axis(image_path, axis='x')`

Calibrate an axis by selecting start and end points.

**Parameters:**
- `image_path` (str): Path to the image file
- `axis` (str): 'x' or 'y' to specify which axis to calibrate

**Returns:**
- `dict`: Calibration parameters including start/end pixels, values, units, and origin coordinates
- Returns `None` if cancelled

### `collect_coordinates(image_path, x_calibration=None, y_calibration=None, save_csv=True)`

Interactively select coordinates from an image.

**Parameters:**
- `image_path` (str): Path to the image file
- `x_calibration` (dict, optional): Calibration parameters from `calibrate_axis(axis='x')`
- `y_calibration` (dict, optional): Calibration parameters from `calibrate_axis(axis='y')`
- `save_csv` (bool, optional): Whether to save individual CSV file. Default `True`.

**Returns:**
- `pandas.DataFrame`: DataFrame with selected coordinates (calibrated if calibrations provided)

**Output:**
- CSV file: `{image_name}_coordinates.csv` (if `save_csv=True`)

## File Structure

```
literature_data/
├── main.py                    # Main script with example usage
├── collect_coordinates.py      # Core functions for calibration and coordinate collection
├── give_paths.py             # Path configuration utility
├── environment.yml           # Conda/mamba environment specification
└── README.md                 # This file
```

## Environment Details

The `lit_data` environment includes:
- **Python 3.11**: Latest stable Python version
- **matplotlib**: For image display and interactive plotting
- **numpy**: For numerical operations
- **pandas**: For data manipulation and CSV export
- **pillow**: For image I/O support
- **tk**: Required for TkAgg backend (interactive windows)
- **opencv-python**: For additional image processing capabilities (via pip)

## Example Workflow

### Batch Processing Workflow

1. Organize your images in a directory
2. Update the directory path in `main.py`
3. Run `python main.py`
4. For each image:
   - Calibrate x-axis (select start/end, enter values and units)
   - Calibrate y-axis (select start/end, enter values and units)
   - Enter body length in cm
   - Collect coordinate points by clicking
5. Find `all_panels_coordinates.csv` with all data

### Single Image Workflow

1. Prepare your image file (PNG, JPG, TIFF, etc.)
2. Calibrate axes if needed
3. Collect coordinates interactively
4. Use the CSV file for analysis

## Tips

- **Calibration accuracy**: Make sure to select the exact start and end points of your axes for accurate calibration
- **Units consistency**: Use consistent units across all images (e.g., always use 'cm')
- **Body length**: Measure or estimate body length accurately for proper scaling
- **Point selection**: You can select points in any order - they will be numbered sequentially
- **Point removal**: Use right-click to remove points if you make mistakes
- **Coordinate system**: Pixel coordinates use image coordinates (origin at top-left)
- **Batch processing**: The `all_panels()` function processes images in alphabetical order

## Troubleshooting

- **Image not displaying**: 
  - Check that the image path is correct and the file format is supported (PNG, JPG, TIFF, etc.)
  - Ensure the image file exists and is readable

- **Interactive window not appearing**:
  - Make sure you're running from a terminal (not just in an IDE)
  - Check that the TkAgg backend is working: `python -c "import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt; plt.figure(); plt.show()"`

- **Environment issues**: 
  - Try recreating the environment: `mamba env create -f environment.yml --force`
  - Activate the environment: `mamba activate lit_data`

- **Import errors**: 
  - Ensure the environment is activated: `mamba activate lit_data`
  - Verify all packages are installed: `python -c "import matplotlib, numpy, pandas"`

- **Calibration not working**:
  - Make sure you click both start and end points before pressing Enter
  - Check that you're entering valid numeric values for calibration

- **CSV file not created**:
  - Check file permissions in the output directory
  - Ensure you pressed Enter or closed the window to finish
  - Verify that at least one coordinate was selected

## Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- TIFF (.tiff, .tif)
- BMP (.bmp)
- GIF (.gif)

Case-insensitive matching (e.g., `.PNG`, `.JPG` are also supported)
