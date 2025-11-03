import matplotlib
# Set interactive backend before importing pyplot
# Try multiple backends in order of preference
backend_options = ['TkAgg', 'Qt5Agg', 'MacOSX', 'QtAgg']
for backend in backend_options:
    try:
        matplotlib.use(backend)
        # Verify it was set
        if matplotlib.get_backend() == backend:
            break
    except Exception:
        continue

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os
from pathlib import Path


def all_panels(image_dir_path):
    """
    Process all image files in a directory by calibrating axes and collecting coordinates.
    All calibration constants and coordinates are saved to a single CSV file.
    
    Args:
        image_dir_path (str): Path to directory containing image files
        
    Returns:
        pandas.DataFrame: Combined DataFrame with all calibration constants and coordinates
    """
    # Supported image file extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF', '.BMP', '.GIF'}
    
    # Get all image files in the directory
    image_dir = Path(image_dir_path)
    if not image_dir.exists():
        raise ValueError(f"Directory does not exist: {image_dir_path}")
    
    image_files = [f for f in image_dir.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    # Sort for consistent processing order
    image_files.sort()
    
    num_images = len(image_files)
    print(f"\nFound {num_images} image file(s) in {image_dir_path}")
    
    if num_images == 0:
        print("No image files found in the specified directory.")
        return pd.DataFrame()
    
    # List to store all combined data
    all_rows = []
    
    # Process each image file
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing image {idx}/{num_images}: {image_path.name}")
        print(f"{'='*60}")
        
        # Calibrate both axes once for this image (shared across all individuals)
        print(f'\n--- Axis Calibration for {image_path.name} ---')
        print('Calibrating axes (this will be used for all individuals in this image)...')
        x_calibration = calibrate_axis(str(image_path), axis='x')
        y_calibration = calibrate_axis(str(image_path), axis='y')
        
        # Prompt for number of individuals
        print(f'\n--- Number of Individuals ---')
        try:
            num_individuals = int(input(f'How many individuals are in {image_path.name}? '))
            if num_individuals < 1:
                print('Number of individuals must be at least 1. Setting to 1.')
                num_individuals = 1
        except (ValueError, KeyboardInterrupt):
            print('Invalid input. Assuming 1 individual.')
            num_individuals = 1
        
        # Process each individual
        for individual_idx in range(1, num_individuals + 1):
            print(f'\n--- Individual {individual_idx}/{num_individuals} ---')
            
            # Prompt for body length for this individual
            print(f'\n--- Body Length Information ---')
            try:
                body_length_cm = float(input(f'Enter the body length (cm) for individual {individual_idx} in {image_path.name}: '))
            except (ValueError, KeyboardInterrupt):
                print('Body length not provided, using None.')
                body_length_cm = None
            
            # Collect coordinates interactively (don't save individual CSV)
            print(f'\nCollecting coordinates for individual {individual_idx}...')
            coordinates_df = collect_coordinates(str(image_path), 
                                                x_calibration=x_calibration,
                                                y_calibration=y_calibration,
                                                save_csv=False)
            
            # Add calibration constants, image filename, and body length to each coordinate row
            if not coordinates_df.empty:
                coordinates_df['image_filename'] = image_path.name
                coordinates_df['individual_number'] = individual_idx
                coordinates_df['body_length_cm'] = body_length_cm
                
                # Add x-axis calibration constants
                if x_calibration is not None:
                    coordinates_df['x_cal_start_pixel'] = x_calibration['start_pixel']
                    coordinates_df['x_cal_end_pixel'] = x_calibration['end_pixel']
                    coordinates_df['x_cal_start_value'] = x_calibration['start_value']
                    coordinates_df['x_cal_end_value'] = x_calibration['end_value']
                    coordinates_df['x_cal_units'] = x_calibration.get('units', '')
                    coordinates_df['x_origin_pixel_x'] = x_calibration['origin_pixel'][0]
                    coordinates_df['x_origin_pixel_y'] = x_calibration['origin_pixel'][1]
                else:
                    coordinates_df['x_cal_start_pixel'] = None
                    coordinates_df['x_cal_end_pixel'] = None
                    coordinates_df['x_cal_start_value'] = None
                    coordinates_df['x_cal_end_value'] = None
                    coordinates_df['x_cal_units'] = None
                    coordinates_df['x_origin_pixel_x'] = None
                    coordinates_df['x_origin_pixel_y'] = None
                
                # Add y-axis calibration constants
                if y_calibration is not None:
                    coordinates_df['y_cal_start_pixel'] = y_calibration['start_pixel']
                    coordinates_df['y_cal_end_pixel'] = y_calibration['end_pixel']
                    coordinates_df['y_cal_start_value'] = y_calibration['start_value']
                    coordinates_df['y_cal_end_value'] = y_calibration['end_value']
                    coordinates_df['y_cal_units'] = y_calibration.get('units', '')
                    coordinates_df['y_origin_pixel_x'] = y_calibration['origin_pixel'][0]
                    coordinates_df['y_origin_pixel_y'] = y_calibration['origin_pixel'][1]
                else:
                    coordinates_df['y_cal_start_pixel'] = None
                    coordinates_df['y_cal_end_pixel'] = None
                    coordinates_df['y_cal_start_value'] = None
                    coordinates_df['y_cal_end_value'] = None
                    coordinates_df['y_cal_units'] = None
                    coordinates_df['y_origin_pixel_x'] = None
                    coordinates_df['y_origin_pixel_y'] = None
                
                all_rows.append(coordinates_df)
            
            print(f"\nCompleted individual {individual_idx}/{num_individuals}")
        
        print(f"\nCompleted processing: {image_path.name}")
    
    # Combine all data into a single DataFrame
    if all_rows:
        combined_df = pd.concat(all_rows, ignore_index=True)
        
        # Reorder columns for better readability
        column_order = [
            'image_filename',
            'body_length_cm',
            'x',
            'y',
            'x_cal_units',
            'y_cal_units',
            'individual_number',
            'x_cal_start_pixel', 'x_cal_end_pixel', 'x_cal_start_value', 'x_cal_end_value',
            'x_origin_pixel_x', 'x_origin_pixel_y',
            'y_cal_start_pixel', 'y_cal_end_pixel', 'y_cal_start_value', 'y_cal_end_value',
            'y_origin_pixel_x', 'y_origin_pixel_y',
            'x_pixel', 'y_pixel'
        ]
        # Only include columns that exist
        column_order = [col for col in column_order if col in combined_df.columns]
        # Add any remaining columns
        remaining_cols = [col for col in combined_df.columns if col not in column_order]
        combined_df = combined_df[column_order + remaining_cols]
        
        # Save to single CSV file in the directory
        output_csv_path = image_dir / 'all_panels_coordinates.csv'
        combined_df.to_csv(output_csv_path, index=False)
        print(f"\n{'='*60}")
        print(f"Finished processing all {num_images} image(s)")
        print(f"Saved all calibration constants and coordinates to: {output_csv_path}")
        print(f"Total coordinates: {len(combined_df)}")
        print(f"{'='*60}")
        
        return combined_df
    else:
        print(f"\n{'='*60}")
        print(f"Finished processing all {num_images} image(s)")
        print("No coordinates were collected from any image.")
        print(f"{'='*60}")
        return pd.DataFrame()


def calibrate_axis(image_path, axis='x'):
    """
    Calibrate an axis (x or y) by selecting start and end points.
    
    Args:
        image_path (str): Path to the image file
        axis (str): 'x' or 'y' to specify which axis to calibrate
        
    Returns:
        dict: Dictionary with calibration parameters:
            - 'axis': 'x' or 'y'
            - 'start_pixel': coordinate of start point in pixels
            - 'end_pixel': coordinate of end point in pixels
            - 'start_value': real-world value at start point
            - 'end_value': real-world value at end point
            - 'units': units for the axis (e.g., 'cm', 'mm', 'm')
            - 'origin_pixel': full (x, y) pixel coordinates of the start point (axis origin)
            Returns None if cancelled
    """
    if axis not in ['x', 'y']:
        raise ValueError("axis must be 'x' or 'y'")
    
    axis_label = axis.upper()
    # Enable interactive mode
    plt.ion()
    
    # Load the image
    img = plt.imread(image_path)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(f'Click on the START point of the {axis_label}-axis')
    ax.axis('off')
    
    # Bring window to front (if supported by backend)
    try:
        fig.canvas.manager.window.raise_()
        fig.canvas.manager.window.attributes('-topmost', True)
        fig.canvas.manager.window.attributes('-topmost', False)
    except (AttributeError, Exception):
        pass
    
    calibration_points = []
    circles = []
    text_labels = []
    closed = False
    current_step = 'start'  # 'start' or 'end'
    
    def on_close(event):
        """Handle window close event"""
        nonlocal closed
        closed = True
    
    def onclick(event):
        """Handle mouse clicks"""
        nonlocal current_step  # Declare at function start
        
        if event.inaxes != ax:
            return
        
        if event.button == 1:  # Left click
            x, y = event.xdata, event.ydata
            
            if current_step == 'start':
                # Clear any previous points
                for circle in circles:
                    circle.remove()
                for label in text_labels:
                    label.remove()
                circles.clear()
                text_labels.clear()
                calibration_points.clear()
                
                calibration_points.append((x, y))
                
                # Draw blue circle for start point
                circle = patches.Circle((x, y), radius=8, color='blue', 
                                       fill=False, linewidth=3)
                ax.add_patch(circle)
                circles.append(circle)
                
                # Add text label
                text_label = ax.text(x + 15, y + 15, 'START', 
                           color='blue', fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
                text_labels.append(text_label)
                
                # Automatically transition to end point selection
                current_step = 'end'
                
                ax.set_title(f'Start point selected. Click on the END point of the {axis_label}-axis, then press "e" to continue')
                fig.canvas.draw()
                print(f'Start point selected: ({x:.2f}, {y:.2f})')
                print(f'Ready for end point selection. Click on the end point of the {axis_label}-axis.')
            
            elif current_step == 'end':
                if len(calibration_points) == 1:
                    calibration_points.append((x, y))
                    
                    # Draw red circle for end point
                    circle = patches.Circle((x, y), radius=8, color='red', 
                                           fill=False, linewidth=3)
                    ax.add_patch(circle)
                    circles.append(circle)
                    
                    # Draw line connecting the two points
                    ax.plot([calibration_points[0][0], calibration_points[1][0]],
                           [calibration_points[0][1], calibration_points[1][1]],
                           'g-', linewidth=2, alpha=0.7)
                    
                    # Add text label
                    text_label = ax.text(x + 15, y + 15, 'END', 
                               color='red', fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
                    text_labels.append(text_label)
                    
                    ax.set_title('End point selected. Press Enter to confirm and enter values')
                    fig.canvas.draw()
                    print(f'End point selected: ({x:.2f}, {y:.2f})')
    
    def on_key(event):
        """Handle key presses"""
        nonlocal closed
        
        # Enter/Return key to finish calibration
        if event.key in ['\n', 'enter', 'return'] and current_step == 'end' and len(calibration_points) == 2:
            # Close the window and proceed to value input
            closed = True
            plt.close(fig)
        
        elif event.key == 'q':
            # Cancel calibration
            closed = True
            plt.close(fig)
            return None
    
    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Show the plot
    plt.tight_layout()
    plt.draw()
    
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        pass
    finally:
        if not closed:
            try:
                plt.close(fig)
            except:
                pass
        plt.ioff()
    
    # Check if calibration was completed
    if len(calibration_points) != 2:
        print('\nCalibration cancelled or incomplete.')
        return None
    
    # Get the pixel coordinate based on axis
    if axis == 'x':
        start_pixel = calibration_points[0][0]
        end_pixel = calibration_points[1][0]
    else:  # y-axis
        start_pixel = calibration_points[0][1]
        end_pixel = calibration_points[1][1]
    
    # Prompt for real-world values and units
    print(f'\n--- {axis_label}-axis Calibration Values ---')
    try:
        start_value = float(input(f'Enter the {axis_label}-axis value at the START point: '))
        end_value = float(input(f'Enter the {axis_label}-axis value at the END point: '))
        units = input(f'Enter the units for the {axis_label}-axis (e.g., BL, BL/s, Hz, cm, mm, m): ').strip()
        if not units:
            units = 'units'  # Default if empty
    except (ValueError, KeyboardInterrupt):
        print('\nCalibration cancelled.')
        return None
    
    # Get the origin pixel coordinates (start point has both x and y)
    origin_pixel_x = calibration_points[0][0]
    origin_pixel_y = calibration_points[0][1]
    
    # Return calibration parameters
    calibration = {
        'axis': axis,
        'start_pixel': start_pixel,
        'end_pixel': end_pixel,
        'start_value': start_value,
        'end_value': end_value,
        'units': units,
        'origin_pixel': (origin_pixel_x, origin_pixel_y)
    }
    
    print(f'\n{axis_label}-axis calibration complete:')
    print(f'  Start: pixel {calibration["start_pixel"]:.2f} = value {calibration["start_value"]} {units}')
    print(f'  End: pixel {calibration["end_pixel"]:.2f} = value {calibration["end_value"]} {units}')
    
    return calibration


def apply_calibration(pixel_value, calibration):
    """
    Convert pixel coordinate to calibrated value.
    
    Args:
        pixel_value (float): coordinate in pixels
        calibration (dict): Calibration parameters from calibrate_axis()
        
    Returns:
        float: Calibrated value
    """
    if calibration is None:
        return pixel_value
    
    start_pixel = calibration['start_pixel']
    end_pixel = calibration['end_pixel']
    start_value = calibration['start_value']
    end_value = calibration['end_value']
    
    # Linear interpolation
    if end_pixel == start_pixel:
        return start_value
    
    ratio = (pixel_value - start_pixel) / (end_pixel - start_pixel)
    calibrated_value = start_value + ratio * (end_value - start_value)
    
    return calibrated_value


def collect_coordinates(image_path, x_calibration=None, y_calibration=None, save_csv=True):
    """
    Interactively select coordinates from an image.
    
    Args:
        image_path (str): Path to the image file
        x_calibration (dict, optional): Calibration parameters from calibrate_axis(axis='x').
                                        If provided, x-coordinates will be calibrated.
        y_calibration (dict, optional): Calibration parameters from calibrate_axis(axis='y').
                                       If provided, y-coordinates will be calibrated.
        save_csv (bool, optional): Whether to save individual CSV file. Default True.
        
    Returns:
        pandas.DataFrame: DataFrame containing the selected coordinates.
                         If calibrations are provided, coordinates will be calibrated.
        
    Usage:
        - Left click on the image to select a point (red circle will appear)
        - Right click to remove the last point
        - Press Enter or close the window to finish and save coordinates
        - Coordinates are saved as CSV in the same directory as the image (if save_csv=True)
    """
    # Enable interactive mode
    plt.ion()
    
    # Load the image
    img = plt.imread(image_path)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title('Click on points to select coordinates. Press Enter or close window when done.')
    ax.axis('off')
    
    # Bring window to front (if supported by backend)
    try:
        fig.canvas.manager.window.raise_()
        fig.canvas.manager.window.attributes('-topmost', True)
        fig.canvas.manager.window.attributes('-topmost', False)
    except (AttributeError, Exception):
        # Backend might not support window management
        pass
    
    # Store clicked points
    points = []
    circles = []  # Store circle patches for removal
    closed = False  # Flag to track if window is closed
    
    def on_close(event):
        """Handle window close event"""
        nonlocal closed
        closed = True
    
    def onclick(event):
        """Handle mouse clicks"""
        if event.inaxes != ax:
            return
        
        # Left click: add point
        if event.button == 1:
            x, y = event.xdata, event.ydata
            points.append((x, y))
            
            # Draw red circle
            circle = patches.Circle((x, y), radius=5, color='red', 
                                   fill=False, linewidth=2)
            ax.add_patch(circle)
            circles.append(circle)
            
            fig.canvas.draw()
            print(f'Point {len(points)}: ({x:.2f}, {y:.2f})')
        
        # Right click (button 3) or middle click (button 2): remove last point
        elif event.button in [2, 3]:
            if points:
                removed_point = points.pop()
                
                # Remove the last circle
                if circles:
                    circles[-1].remove()
                    circles.pop()
                
                fig.canvas.draw()
                print(f'Removed point: ({removed_point[0]:.2f}, {removed_point[1]:.2f})')
            else:
                print('No points to remove.')
    
    def on_key(event):
        """Handle key presses"""
        # Enter/Return key to finish collecting coordinates
        if event.key in ['\n', 'enter', 'return']:
            nonlocal closed
            if not closed:  # Only close once
                closed = True
                # Close window - this will trigger close_event
                plt.close(fig)
    
    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Show the plot and ensure it blocks
    plt.tight_layout()
    plt.draw()
    
    # Use a more efficient blocking approach
    try:
        plt.show(block=True)  # Block until window is closed
    except KeyboardInterrupt:
        pass
    finally:
        # Quick cleanup - just disable interactive mode
        # Don't close again if already closed
        if not closed:
            try:
                plt.close(fig)
            except:
                pass
        plt.ioff()
    
    # Create DataFrame with coordinates
    if points:
        # Apply calibrations if provided
        x_calibrated = [apply_calibration(x, x_calibration) for x, y in points]
        y_calibrated = [apply_calibration(y, y_calibration) for x, y in points]
        
        # Build DataFrame with calibrated values and original pixel values
        df_data = {
            'x': x_calibrated,
            'y': y_calibrated,
            'x_pixel': [x for x, y in points],
            'y_pixel': [y for x, y in points]
        }
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV in the same directory as the image (if requested)
        if save_csv:
            image_dir = Path(image_path).parent
            image_name = Path(image_path).stem
            csv_path = image_dir / f'{image_name}_coordinates.csv'
            
            df.to_csv(csv_path, index=False)
            print(f'\nSaved {len(points)} coordinates to: {csv_path}')
        
        calibration_notes = []
        if x_calibration is not None:
            calibration_notes.append('x-coordinates are calibrated')
        if y_calibration is not None:
            calibration_notes.append('y-coordinates are calibrated')
        if calibration_notes and save_csv:
            print(f'Note: {", ".join(calibration_notes)}. Original pixel values saved as x_pixel and y_pixel.')
        
        return df
    else:
        print('\nNo coordinates were selected.')
        return pd.DataFrame(columns=['x', 'y'])

