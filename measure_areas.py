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
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import os
from pathlib import Path


def select_bezier_landmarks(image_path, shape_name=None):
    """
    Interactively select control points for a bezier curve that defines a shape.
    
    Args:
        image_path (str): Path to the image file
        shape_name (str, optional): Name/identifier for this shape (e.g., 'body', 'fin')
        
    Returns:
        dict: Dictionary containing:
            - 'control_points': List of (x, y) tuples for control points
            - 'curve_points': List of (x, y) tuples for the bezier curve
            - 'shape_name': Name of the shape
            - 'image_path': Path to the image
            Returns None if cancelled
    """
    # Load the image
    img = plt.imread(image_path)
    
    # Create figure and axis (don't use ion() to avoid auto-display issues)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    # Create title with shape name if provided
    if shape_name:
        title = f'Selecting control points for: {shape_name}\nClick to add control points. Press Enter when done (minimum 3 points).\nPress ESC in this window or Ctrl+C in terminal to cancel.'
    else:
        title = 'Click to add control points for bezier curve. Press Enter when done (minimum 3 points).\nPress ESC in this window or Ctrl+C in terminal to cancel.'
    
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    
    # Bring window to front (if supported by backend)
    try:
        fig.canvas.manager.window.raise_()
        fig.canvas.manager.window.attributes('-topmost', True)
        fig.canvas.manager.window.attributes('-topmost', False)
    except (AttributeError, Exception):
        pass
    
    # Store control points
    control_points = []
    circles = []  # Store circle patches for removal
    curve_line = None  # Store the curve line for removal
    control_line = None  # Store the control point connection line for removal
    closed = False  # Flag to track if window is closed
    
    def on_close(event):
        """Handle window close event"""
        nonlocal closed
        closed = True
    
    def update_curve():
        """Update the bezier curve visualization"""
        nonlocal curve_line, control_line
        
        # Remove old curve if it exists
        if curve_line is not None:
            curve_line.remove()
            curve_line = None
        
        # Remove old control point connection line if it exists
        if control_line is not None:
            control_line.remove()
            control_line = None
        
        # Need at least 3 points for a bezier curve
        if len(control_points) >= 3:
            # Generate bezier curve points
            curve_points = generate_bezier_curve(control_points, num_points=200)
            
            # Draw the curve
            curve_x = [p[0] for p in curve_points]
            curve_y = [p[1] for p in curve_points]
            curve_line, = ax.plot(curve_x, curve_y, 'b-', linewidth=2, alpha=0.7, label='Bezier Curve')
            
            # Draw control point connections (for visualization)
            if len(control_points) > 1:
                cp_x = [p[0] for p in control_points]
                cp_y = [p[1] for p in control_points]
                control_line, = ax.plot(cp_x, cp_y, 'g--', linewidth=1, alpha=0.3)
            
            fig.canvas.draw()
    
    def onclick(event):
        """Handle mouse clicks"""
        if event.inaxes != ax:
            return
        
        # Left click: add control point
        if event.button == 1:
            x, y = event.xdata, event.ydata
            control_points.append((x, y))
            
            # Draw blue circle for control point
            circle = patches.Circle((x, y), radius=6, color='blue', 
                                   fill=True, linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            circles.append(circle)
            
            # Add text label with point number
            text_label = ax.text(x + 10, y + 10, str(len(control_points)), 
                               color='blue', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            circles.append(text_label)  # Store text as well for removal
            
            # Update curve
            update_curve()
            
            fig.canvas.draw()
        
        # Right click (button 3) or middle click (button 2): remove last point
        elif event.button in [2, 3]:
            if control_points:
                removed_point = control_points.pop()
                
                # Remove the last two items (circle and text)
                if len(circles) >= 2:
                    circles[-1].remove()
                    circles.pop()
                    circles[-1].remove()
                    circles.pop()
                elif len(circles) == 1:
                    circles[-1].remove()
                    circles.pop()
                
                # Update curve
                update_curve()
                
                fig.canvas.draw()
    
    cancelled = False  # Flag to track if ESC was pressed
    
    def on_key(event):
        """Handle key presses"""
        nonlocal closed, cancelled
        # Enter/Return key to finish collecting points
        if event.key in ['\n', 'enter', 'return']:
            if len(control_points) < 3:
                return
            if not closed:  # Only close once
                closed = True
                plt.close(fig)
        # ESC key to cancel and exit (check multiple possible key names and formats)
        elif (event.key in ['escape', 'esc'] or 
              str(event.key).lower() in ['escape', 'esc'] or
              (hasattr(event, 'key') and 'escape' in str(event.key).lower())):
            cancelled = True
            if not closed:
                closed = True
                plt.close(fig)
                # Force stop the event loop
                try:
                    fig.canvas.stop_event_loop()
                except:
                    pass
    
    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Show the plot (only once)
    plt.tight_layout()
    plt.draw()
    
    # Make sure figure can receive keyboard events and has focus
    try:
        fig.canvas.set_window_title('Select Control Points')
        # Ensure figure has focus for keyboard events
        fig.canvas.manager.window.activateWindow()
        fig.canvas.manager.window.raise_()
    except:
        pass
    
    try:
        # Use a blocking show that waits for window to close
        # Don't call fig.show() separately to avoid duplicate display
        plt.show(block=True)
    except KeyboardInterrupt:
        cancelled = True
        closed = True
    finally:
        if not closed:
            try:
                plt.close(fig)
            except:
                pass
        # Ensure we're not in interactive mode to prevent duplicate displays
        plt.ioff()
    
    # Check if cancelled via ESC
    if cancelled:
        return 'CANCELLED'
    
    # Check if we have enough points
    if len(control_points) < 3:
        return None
    
    # Generate the final bezier curve (reduced points for faster generation)
    curve_points = generate_bezier_curve(control_points, num_points=300)
    
    result = {
        'control_points': control_points,
        'curve_points': curve_points,
        'shape_name': shape_name,
        'image_path': image_path
    }
    
    return result


def cubic_bezier(p0, p1, p2, p3, t):
    """
    Calculate a point on a cubic bezier curve.
    
    Args:
        p0, p1, p2, p3: Control points as (x, y) tuples
        t: Parameter value between 0 and 1
        
    Returns:
        (x, y) tuple representing a point on the curve
    """
    t = np.clip(t, 0, 1)
    mt = 1 - t
    mt2 = mt * mt
    t2 = t * t
    
    x = mt2 * mt * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t2 * t * p3[0]
    y = mt2 * mt * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t2 * t * p3[1]
    
    return (x, y)


def catmull_rom(p0, p1, p2, p3, t):
    """
    Calculate a point on a Catmull-Rom spline segment.
    The curve passes through p1 and p2, with tangents determined by p0 and p3.
    This ensures C1 continuity (smooth curves with no kinks).
    
    Args:
        p0, p1, p2, p3: Control points as (x, y) tuples or numpy arrays
        t: Parameter value between 0 and 1 (0 = p1, 1 = p2)
        
    Returns:
        (x, y) tuple representing a point on the curve
    """
    t = np.clip(t, 0, 1)
    t2 = t * t
    t3 = t2 * t
    
    # Catmull-Rom spline formula
    # P(t) = 0.5 * [(-t^3 + 2t^2 - t)P0 + (3t^3 - 5t^2 + 2)P1 + (-3t^3 + 4t^2 + t)P2 + (t^3 - t^2)P3]
    x = 0.5 * ((-t3 + 2*t2 - t) * p0[0] + 
               (3*t3 - 5*t2 + 2) * p1[0] + 
               (-3*t3 + 4*t2 + t) * p2[0] + 
               (t3 - t2) * p3[0])
    y = 0.5 * ((-t3 + 2*t2 - t) * p0[1] + 
               (3*t3 - 5*t2 + 2) * p1[1] + 
               (-3*t3 + 4*t2 + t) * p2[1] + 
               (t3 - t2) * p3[1])
    
    return (x, y)


def generate_bezier_curve(control_points, num_points=200):
    """
    Generate points along a smooth curve from control points.
    Uses Catmull-Rom splines for 5+ points (ensures C1 continuity, no kinks),
    cubic bezier for 4 points, and quadratic bezier for 3 points.
    
    Args:
        control_points (list): List of (x, y) tuples for control points
        num_points (int): Number of points to generate along the curve
        
    Returns:
        list: List of (x, y) tuples representing points on the curve
    """
    if len(control_points) < 2:
        return control_points
    
    if len(control_points) == 2:
        # Linear interpolation for 2 points
        t = np.linspace(0, 1, num_points)
        x = np.interp(t, [0, 1], [control_points[0][0], control_points[1][0]])
        y = np.interp(t, [0, 1], [control_points[0][1], control_points[1][1]])
        return list(zip(x, y))
    
    # Convert to numpy array
    points = np.array(control_points)
    
    # Use Catmull-Rom splines for smooth curves through all control points
    # Catmull-Rom splines are C1 continuous (smooth, no kinks) and pass through all control points
    curve_points = []
    
    if len(control_points) == 3:
        # For 3 points, use quadratic bezier (smooth through all points, vectorized)
        t = np.linspace(0, 1, num_points)
        mt = 1 - t
        t2 = t * t
        
        # Vectorized quadratic bezier calculation
        x = mt * mt * points[0][0] + 2 * mt * t * points[1][0] + t2 * points[2][0]
        y = mt * mt * points[0][1] + 2 * mt * t * points[1][1] + t2 * points[2][1]
        
        curve_points = list(zip(x, y))
    elif len(control_points) == 4:
        # For exactly 4 points, use cubic bezier (vectorized for performance)
        t = np.linspace(0, 1, num_points)
        mt = 1 - t
        mt2 = mt * mt
        t2 = t * t
        t3 = t2 * t
        
        # Vectorized cubic bezier calculation
        x = mt2 * mt * points[0][0] + 3 * mt2 * t * points[1][0] + 3 * mt * t2 * points[2][0] + t3 * points[3][0]
        y = mt2 * mt * points[0][1] + 3 * mt2 * t * points[1][1] + 3 * mt * t2 * points[2][1] + t3 * points[3][1]
        
        curve_points = list(zip(x, y))
    else:
        # For 5+ points, use Catmull-Rom spline for smooth, kink-free curves
        # Catmull-Rom passes through all control points with C1 continuity
        
        # Add virtual endpoints by duplicating first and last points
        # This ensures the curve starts at the first point and ends at the last point
        extended_points = np.vstack([points[0:1], points, points[-1:]])
        
        # Number of segments (one segment between each pair of control points)
        num_segments = len(control_points) - 1
        points_per_segment = max(10, num_points // max(1, num_segments))
        
        # Vectorized Catmull-Rom spline generation for better performance
        all_segment_points = []
        
        for i in range(num_segments):
            # Each segment uses 4 points: [i, i+1, i+2, i+3] in extended_points
            # The segment goes from point i+1 to point i+2
            p0 = extended_points[i]
            p1 = extended_points[i + 1]  # Start of segment
            p2 = extended_points[i + 2]  # End of segment
            p3 = extended_points[i + 3]
            
            # Generate t values for this segment
            t_seg = np.linspace(0, 1, points_per_segment)
            
            # Vectorized Catmull-Rom calculation
            t2 = t_seg * t_seg
            t3 = t2 * t_seg
            
            # Catmull-Rom coefficients
            c0 = 0.5 * (-t3 + 2*t2 - t_seg)
            c1 = 0.5 * (3*t3 - 5*t2 + 2)
            c2 = 0.5 * (-3*t3 + 4*t2 + t_seg)
            c3 = 0.5 * (t3 - t2)
            
            # Calculate all points for this segment at once
            x_seg = c0 * p0[0] + c1 * p1[0] + c2 * p2[0] + c3 * p3[0]
            y_seg = c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1]
            
            # Stack x and y coordinates
            segment_points = np.column_stack([x_seg, y_seg])
            all_segment_points.append(segment_points)
        
        # Concatenate all segments
        if all_segment_points:
            curve_array = np.vstack(all_segment_points)
        else:
            curve_array = np.array([]).reshape(0, 2)
        
        # Ensure we have exactly num_points by resampling if needed
        if len(curve_array) != num_points:
            if len(curve_array) == 0:
                # Edge case: no points generated
                curve_points = [tuple(points[0])] * num_points
            else:
                # Parameterize by arc length (vectorized)
                diffs = np.diff(curve_array, axis=0)
                segment_lengths = np.linalg.norm(diffs, axis=1)
                distances = np.concatenate(([0], np.cumsum(segment_lengths)))
                
                if distances[-1] > 0:
                    distances = distances / distances[-1]
                    # Resample to get exactly num_points
                    t_new = np.linspace(0, 1, num_points)
                    
                    # Handle potential duplicate values in distances array
                    # np.interp requires strictly increasing x values
                    unique_mask = np.concatenate(([True], np.diff(distances) > 1e-10))
                    unique_distances = distances[unique_mask]
                    unique_points = curve_array[unique_mask]
                    
                    # If we have at least 2 unique points, interpolate
                    if len(unique_distances) >= 2:
                        # Ensure unique_distances is strictly increasing
                        if unique_distances[0] == unique_distances[-1]:
                            # All points collapsed to one - return identical points
                            curve_points = [tuple(curve_array[0])] * num_points
                        else:
                            x_new = np.interp(t_new, unique_distances, unique_points[:, 0])
                            y_new = np.interp(t_new, unique_distances, unique_points[:, 1])
                            curve_points = list(zip(x_new, y_new))
                    else:
                        # Only one unique point - return identical points
                        curve_points = [tuple(curve_array[0])] * num_points
                else:
                    # All points are identical (zero total distance)
                    # Return list of identical points with correct length
                    curve_points = [tuple(curve_array[0])] * num_points
        else:
            # Convert numpy array to list of tuples
            curve_points = [tuple(pt) for pt in curve_array]
    
    return curve_points


def calculate_area(curve_points):
    """
    Calculate the area enclosed by a closed curve using the shoelace formula.
    
    Args:
        curve_points (list): List of (x, y) tuples representing points on the curve
        
    Returns:
        float: Area enclosed by the curve (in pixel^2 or calibrated units^2)
    """
    if len(curve_points) < 3:
        return 0.0
    
    # Convert to numpy array
    points = np.array(curve_points)
    
    # Ensure the curve is closed (first point == last point)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0:1]])
    
    # Shoelace formula
    x = points[:, 0]
    y = points[:, 1]
    
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    return area


def save_bezier_data(bezier_data, output_dir=None, image_path=None, output_path=None):
    """
    Save bezier curve data to CSV file.
    
    Args:
        bezier_data (dict or list): Dictionary with bezier data, or list of dictionaries
        output_dir (str, optional): Directory to save CSV. If None, uses image directory
        image_path (str, optional): Path to the image file (used if output_dir is None)
        output_path (str or Path, optional): Full path to save CSV file. If provided, 
                                            overrides output_dir and default filename.
        
    Returns:
        str: Path to the saved CSV file
    """
    # If output_path is provided, use it directly
    if output_path is not None:
        csv_path = Path(output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Determine output directory
        if output_dir is None:
            if image_path is None:
                if isinstance(bezier_data, list) and len(bezier_data) > 0:
                    image_path = bezier_data[0].get('image_path')
                elif isinstance(bezier_data, dict):
                    image_path = bezier_data.get('image_path')
            
            if image_path is None:
                raise ValueError("Must provide either output_dir, image_path, or output_path")
            
            output_dir = Path(image_path).parent
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use default filename
        csv_path = output_dir / 'bezier_curves_data.csv'
    
    # Normalize to list
    if isinstance(bezier_data, dict):
        bezier_data = [bezier_data]
    
    # Prepare data for CSV
    rows = []
    for idx, data in enumerate(bezier_data):
        control_points = data['control_points']
        curve_points = data['curve_points']
        shape_name = data.get('shape_name', f'shape_{idx+1}')
        img_path = data.get('image_path', image_path)
        
        # Calculate area
        area = calculate_area(curve_points)
        
        # Create row for each control point
        for cp_idx, (cx, cy) in enumerate(control_points):
            row = {
                'image_path': str(img_path),
                'shape_name': shape_name,
                'shape_index': idx + 1,
                'control_point_index': cp_idx + 1,
                'control_point_x': cx,
                'control_point_y': cy,
                'num_control_points': len(control_points),
                'num_curve_points': len(curve_points),
                'area': area
            }
            rows.append(row)
        
        # Add a summary row with curve points (first few for reference)
        # Store all curve points as a separate entry
        for curve_idx, (curve_x, curve_y) in enumerate(curve_points):
            row = {
                'image_path': str(img_path),
                'shape_name': shape_name,
                'shape_index': idx + 1,
                'control_point_index': None,  # This is a curve point
                'control_point_x': None,
                'control_point_y': None,
                'num_control_points': len(control_points),
                'num_curve_points': len(curve_points),
                'area': area,
                'curve_point_index': curve_idx + 1,
                'curve_point_x': curve_x,
                'curve_point_y': curve_y
            }
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV (csv_path already determined above)
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


def save_bezier_image(bezier_data, output_path=None, image_path=None, show_control_points=True):
    """
    Save an image with the bezier curve overlaid on the original image.
    
    Args:
        bezier_data (dict or list): Dictionary with bezier data, or list of dictionaries
        output_path (str, optional): Path to save the output image. If None, auto-generates
        image_path (str, optional): Path to the original image (used if output_path is None)
        show_control_points (bool): Whether to show control points on the image
        
    Returns:
        str: Path to the saved image file
    """
    # Normalize to list
    if isinstance(bezier_data, dict):
        bezier_data = [bezier_data]
    
    # Determine image path
    if image_path is None:
        if len(bezier_data) > 0:
            image_path = bezier_data[0].get('image_path')
        else:
            raise ValueError("Must provide either output_path or image_path in bezier_data")
    
    # Determine output path
    if output_path is None:
        image_dir = Path(image_path).parent
        image_name = Path(image_path).stem
        output_path = image_dir / f'{image_name}_bezier_traced.jpg'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load the image
    img = plt.imread(image_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')
    
    # Plot each bezier curve
    colors = plt.cm.tab10(np.linspace(0, 1, len(bezier_data)))
    
    for idx, data in enumerate(bezier_data):
        control_points = data['control_points']
        curve_points = data['curve_points']
        shape_name = data.get('shape_name', f'shape_{idx+1}')
        color = colors[idx % len(colors)]
        
        # Plot the curve
        curve_x = [p[0] for p in curve_points]
        curve_y = [p[1] for p in curve_points]
        ax.plot(curve_x, curve_y, color=color, linewidth=3, alpha=0.8, 
               label=shape_name if shape_name else f'Shape {idx+1}')
        
        # Plot control points if requested
        if show_control_points:
            cp_x = [p[0] for p in control_points]
            cp_y = [p[1] for p in control_points]
            ax.scatter(cp_x, cp_y, color=color, s=100, marker='o', 
                      edgecolors='white', linewidths=2, alpha=0.9, zorder=5)
            
            # Draw lines connecting control points
            ax.plot(cp_x, cp_y, color=color, linestyle='--', linewidth=1, alpha=0.4)
    
    # Add legend if multiple shapes
    if len(bezier_data) > 1:
        ax.legend(loc='upper right', fontsize=10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    return str(output_path)


def measure_shape_area(image_path, shape_name=None, output_dir=None):
    """
    Complete workflow: Select landmarks, create bezier curve, calculate area, and save results.
    
    Args:
        image_path (str): Path to the image file
        shape_name (str, optional): Name/identifier for this shape
        output_dir (str, optional): Directory to save outputs. If None, uses image directory
        
    Returns:
        dict: Dictionary containing:
            - 'bezier_data': Dictionary with control points, curve points, etc.
            - 'area': Calculated area
            - 'csv_path': Path to saved CSV file
            - 'image_path': Path to saved image file
    """
    # Select landmarks and create bezier curve
    bezier_data = select_bezier_landmarks(image_path, shape_name=shape_name)
    
    # Check if cancelled
    if bezier_data == 'CANCELLED':
        return 'CANCELLED'
    
    if bezier_data is None:
        return None
    
    # Calculate area
    area = calculate_area(bezier_data['curve_points'])
    bezier_data['area'] = area
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(image_path).parent
    else:
        output_dir = Path(output_dir)
    
    # Create filename-safe shape name (remove special characters)
    safe_shape_name = shape_name.replace(' ', '_').replace('/', '_').replace('\\', '_') if shape_name else 'shape'
    
    # Save CSV with shape name in filename
    csv_filename = f'{Path(image_path).stem}_{safe_shape_name}_bezier_data.csv'
    csv_path = output_dir / csv_filename
    csv_path_saved = save_bezier_data(bezier_data, output_path=csv_path, image_path=image_path)
    
    # Save image with shape name in filename
    image_output_path = output_dir / f'{Path(image_path).stem}_{safe_shape_name}_bezier_traced.jpg'
    image_path_saved = save_bezier_image(bezier_data, output_path=image_output_path, 
                                        image_path=image_path)
    
    result = {
        'bezier_data': bezier_data,
        'area': area,
        'csv_path': csv_path_saved,
        'image_path': image_path_saved,
        'shape_name': shape_name
    }
    
    return result


def all_shapes(image_dir_path):
    """
    Process all image files in a directory, allowing user to measure shapes on each image.
    
    Args:
        image_dir_path (str): Path to directory containing image files
        
    Returns:
        pandas.DataFrame: Combined DataFrame with all bezier curve data
    """
    # Supported image file extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', 
                       '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF', '.BMP', '.GIF'}
    
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
    
    # List to store all bezier data
    all_bezier_data = []
    shape_results = []  # Store results for area summary
    cancelled = False  # Flag to track if ESC was pressed
    
    # Process each image file
    for idx, image_path in enumerate(image_files, 1):
        # Check if cancelled before processing next image
        if cancelled:
            break
            
        # Ask how many shapes to measure
        try:
            num_shapes = int(input(f'How many shapes do you want to measure in {image_path.name}? (Press Ctrl+C to cancel): '))
            if num_shapes < 1:
                num_shapes = 1
        except ValueError:
            num_shapes = 1
        except KeyboardInterrupt:
            cancelled = True
            break
        
        # Prompt for all shape names before collecting control points
        shape_names = []
        try:
            for shape_idx in range(1, num_shapes + 1):
                try:
                    shape_name = input(f'Enter a name for shape {shape_idx} (or press Enter for default): ').strip()
                    if not shape_name:
                        shape_name = f'shape_{shape_idx}'
                    shape_names.append(shape_name)
                except ValueError:
                    shape_name = f'shape_{shape_idx}'
                    shape_names.append(shape_name)
        except KeyboardInterrupt:
            cancelled = True
            break
        
        # Process each shape
        result = None
        for shape_idx in range(1, num_shapes + 1):
            # Check if cancelled before processing next shape
            if cancelled:
                break
                
            shape_name = shape_names[shape_idx - 1]
            
            # Measure the shape
            result = measure_shape_area(str(image_path), shape_name=shape_name, 
                                       output_dir=image_dir)
            
            # Check if cancelled (ESC was pressed)
            if result == 'CANCELLED':
                cancelled = True
                break
            
            if result is not None:
                all_bezier_data.append(result['bezier_data'])
                # Store result info for summary
                shape_results.append({
                    'image_name': image_path.name,
                    'shape_name': shape_name,
                    'area': result['area'],
                    'csv_path': result['csv_path'],
                    'image_path': result['image_path']
                })
        
        # Check if we should exit (ESC was pressed)
        if cancelled:
            break
    
    # Print summary of all areas at the end (only if not cancelled)
    if not cancelled and all_bezier_data:
        # Save combined CSV (optional - for reference)
        combined_csv_path = image_dir / 'all_shapes_bezier_data.csv'
        save_bezier_data(all_bezier_data, output_path=combined_csv_path)
        
        # Print area summary
        if shape_results:
            print(f"\n{'='*60}")
            print("AREA SUMMARY")
            print(f"{'='*60}")
            print(f"{'Image':<40} {'Shape Name':<20} {'Area':<20}")
            print(f"{'-'*80}")
            for res in shape_results:
                print(f"{res['image_name']:<40} {res['shape_name']:<20} {res['area']:<20.2f}")
            print(f"{'='*60}")
    elif cancelled and all_bezier_data:
        # Save data collected so far if cancelled
        combined_csv_path = image_dir / 'all_shapes_bezier_data.csv'
        save_bezier_data(all_bezier_data, output_path=combined_csv_path)
    
    # Return a summary DataFrame
    if all_bezier_data:
        summary_data = []
        for data in all_bezier_data:
            summary_data.append({
                'image_path': data.get('image_path', ''),
                'shape_name': data.get('shape_name', ''),
                'num_control_points': len(data['control_points']),
                'num_curve_points': len(data['curve_points']),
                'area': data.get('area', calculate_area(data['curve_points']))
            })
        return pd.DataFrame(summary_data)
    else:
        return pd.DataFrame()

