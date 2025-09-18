#!/usr/bin/env python3
# filepath: /home/panastasiadis/ALP/include/graphblas/cost/cost_model_plot_cg_analysis.py
"""
Script to plot performance data from analysis files produced by the benchmark script.
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.colors as mcolors

# Cache sizes in bytes
CACHE_SIZES = {
    'L1': 64 * 1024,        # 64 KB
    'L2': 512 * 1024,       # 512 KB
    'L3': 24 * 1024 * 1024  # 24 MB
}

def extract_info_from_filename(filename):
    """Extract both matrix size (N) and thread count from the analysis filename."""
    # Default thread count if not specified
    thread_count = 1
    
    # First try the _threads-X pattern (with hyphen)
    match = re.search(r'banded_diag_(\d+)x\d+_band_\d+_threads-(\d+)_analysis\.log', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Then try the _threads_X pattern (with underscore)
    match = re.search(r'banded_diag_(\d+)x\d+_band_\d+_threads_(\d+)_analysis\.log', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Then try the _Xt pattern
    match = re.search(r'banded_diag_(\d+)x\d+_band_\d+_(\d+)t_analysis\.log', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Finally, try without thread specification
    match = re.search(r'banded_diag_(\d+)x\d+_band_\d+_analysis\.log', filename)
    if match:
        return int(match.group(1)), thread_count
    
    # Return None for both if no match
    return None, None

def extract_operator_info(args):
    """Extract operator information from the argument string."""
    # Look for operators
    operators = re.findall(r'operators::(\w+)', args)
    if operators:
        return "_".join(operators)
    return ""

def simplify_args(args):
    """
    Create a simplified representation of args that removes size/dimension information
    but keeps operator and type info.
    """
    # Extract data type
    dtype_match = re.search(r'<(\w+)>', args)
    dtype = dtype_match.group(1) if dtype_match else ""
    
    # Extract operator information
    operator_info = extract_operator_info(args)
    
    # Remove all size-related information
    simplified = re.sub(r'size=\d+', 'size=X', args)
    simplified = re.sub(r'rows=\d+,cols=\d+', 'rows=X,cols=X', simplified)
    simplified = re.sub(r'nnz=\d+', 'nnz=X', simplified)
    
    # Create a key based on type and operator
    if dtype and operator_info:
        return f"{dtype}_{operator_info}"
    elif dtype:
        return dtype
    elif operator_info:
        return operator_info
    else:
        return simplified[:30] + "..." if len(simplified) > 30 else simplified

def parse_memory_footprint(footprint_str):
    """Parse memory footprint string (e.g. '15.0 KB') and convert to bytes."""
    if not footprint_str or footprint_str == "0 B":
        return 0
        
    match = re.match(r'([\d.]+)\s+([KMGTP]?B)', footprint_str)
    if not match:
        return 0
        
    value = float(match.group(1))
    unit = match.group(2)
    
    # Convert to bytes
    if unit == 'KB':
        return value * 1024
    elif unit == 'MB':
        return value * 1024 * 1024
    elif unit == 'GB':
        return value * 1024 * 1024 * 1024
    elif unit == 'TB':
        return value * 1024 * 1024 * 1024 * 1024
    elif unit == 'PB':
        return value * 1024 * 1024 * 1024 * 1024 * 1024
    else:  # Bytes
        return value

def parse_analysis_file(filepath):
    """Parse an analysis file to extract function metrics."""
    result = {}
    current_function = None
    current_args = None
    current_model_data = None
    
    # Get matrix size and thread count from filename
    matrix_size, thread_count = extract_info_from_filename(os.path.basename(filepath))
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Find function analysis sections
            if line.startswith('Analysis for function'):
                match = re.search(r"Analysis for function '(\w+)':", line)
                if match:
                    current_function = match.group(1)
                    result[current_function] = defaultdict(lambda: {'models': []})
            
            # Find argument types
            elif line.startswith('Argument types:') and current_function:
                current_args = line.replace('Argument types:', '').strip()
                # Initialize if not already present
                if current_args not in result[current_function]:
                    result[current_function][current_args] = {
                        'count': 0,
                        'per_iter_count': None,  # New field for per-iteration count
                        'execution_time_min': 0.0,
                        'execution_time_max': 0.0,
                        'execution_time_avg': 0.0,
                        'execution_time_stddev': 0.0,
                        'matrix_size': matrix_size,
                        'thread_count': thread_count,
                        'simplified_args': simplify_args(current_args),
                        'operator_info': extract_operator_info(current_args),
                        'memory_footprint_bytes': 0,
                        'memory_footprint_str': '0 B',
                        'models': []
                    }
                
                # Get invocation count from next line if available
                if i + 1 < len(lines) and lines[i+1].strip().startswith('Invocation count:'):
                    count_line = lines[i+1].strip()
                    
                    # Extract both total count and per-iteration count
                    # Format: "Invocation count: X (per solver iteration: Y)"
                    count_match = re.search(r'Invocation count: (\d+)(?: \(per solver iteration: (\d+)\))?', count_line)
                    if count_match:
                        total_count = int(count_match.group(1))
                        per_iter_count = int(count_match.group(2)) if count_match.group(2) else None
                        
                        result[current_function][current_args]['count'] = total_count
                        result[current_function][current_args]['per_iter_count'] = per_iter_count
                    else:
                        # Fallback to simple parsing if the new format is not found
                        count = int(count_line.replace('Invocation count:', '').strip().split()[0])
                        result[current_function][current_args]['count'] = count
                    
                    i += 1  # Skip the next line since we've processed it
            
            # Find model information
            elif line.startswith('Model:') and current_function and current_args:
                # Start a new model data dictionary
                current_model_data = {
                    'model_name': line.replace('Model:', '').strip(),
                    'threads': None,
                    'aggregator': None,
                    'footprint': None,
                    'level_range': None,
                    'cost': None
                }
                
                # Read the next lines to get model details
                model_lines = []
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('-' * 10):
                    model_lines.append(lines[j].strip())
                    j += 1
                
                # Process model lines
                for model_line in model_lines:
                    if model_line.startswith('Threads:'):
                        current_model_data['threads'] = int(model_line.replace('Threads:', '').strip())
                    elif model_line.startswith('Stream aggregator:'):
                        current_model_data['aggregator'] = model_line.replace('Stream aggregator:', '').strip()
                    elif model_line.startswith('Memory footprint:'):
                        footprint_str = model_line.replace('Memory footprint:', '').strip()
                        current_model_data['footprint'] = footprint_str
                        result[current_function][current_args]['memory_footprint_str'] = footprint_str
                        result[current_function][current_args]['memory_footprint_bytes'] = parse_memory_footprint(footprint_str)
                    elif model_line.startswith('Level range:'):
                        level_range_str = model_line.replace('Level range:', '').strip()
                        try:
                            level_range = eval(level_range_str)  # Safely convert [1, 1] string to list
                            current_model_data['level_range'] = level_range
                        except:
                            current_model_data['level_range'] = [1, 1]  # Default
                    elif model_line.startswith('Predicted cost:'):
                        cost_str = model_line.replace('Predicted cost:', '').strip()
                        current_model_data['cost'] = float(cost_str)
                
                # Add the model data to the result
                if current_model_data:
                    result[current_function][current_args]['models'].append(current_model_data)
                
                # Skip to the end of the model section
                i = j
            
            # Find execution time statistics
            elif line.startswith('Execution time (seconds):') and current_function and current_args:
                # Parse the execution time stats
                match = re.search(r'min=([\d.e+-]+), max=([\d.e+-]+), avg=([\d.e+-]+), std_dev=([\d.e+-]+)', line)
                if match:
                    result[current_function][current_args]['execution_time_min'] = float(match.group(1))
                    result[current_function][current_args]['execution_time_max'] = float(match.group(2))
                    result[current_function][current_args]['execution_time_avg'] = float(match.group(3))
                    result[current_function][current_args]['execution_time_stddev'] = float(match.group(4))
                else:
                    # Handle case where there's only one measurement
                    match = re.search(r'Execution time \(seconds\): ([\d.e+-]+)', line)
                    if match:
                        value = float(match.group(1))
                        result[current_function][current_args]['execution_time_min'] = value
                        result[current_function][current_args]['execution_time_max'] = value
                        result[current_function][current_args]['execution_time_avg'] = value
                        result[current_function][current_args]['execution_time_stddev'] = 0.0
            
            i += 1
    
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return {}
        
    return result

def collect_all_results(results_dir):
    """Collect results from all analysis files in the directory."""
    all_files = glob.glob(os.path.join(results_dir, '*_analysis.log'))
    
    # Organize data by thread count -> function -> simplified_args -> size -> data
    thread_data = {}
    
    for filepath in all_files:
        matrix_size, thread_count = extract_info_from_filename(os.path.basename(filepath))
        
        if matrix_size is None:
            continue
            
        file_results = parse_analysis_file(filepath)
        
        # Skip if file_results is empty
        if not file_results:
            print(f"Warning: No valid data found in {filepath}")
            continue
        
        # Initialize thread count in data structure if needed
        if thread_count not in thread_data:
            thread_data[thread_count] = {}
            
        for function_name, args_data in file_results.items():
            # Initialize function in data structure if needed
            if function_name not in thread_data[thread_count]:
                thread_data[thread_count][function_name] = {}
            
            for args, metrics in args_data.items():
                # Create a key that removes size information
                simplified_args = simplify_args(args)
                
                # Get operator info for display
                operator_info = extract_operator_info(args)
                
                # We don't need to add thread count to the key now as we're already grouping by thread count
                key = simplified_args
                
                # Initialize simplified args in data structure if needed
                if key not in thread_data[thread_count][function_name]:
                    thread_data[thread_count][function_name][key] = {
                        'sizes': {},
                        'operator_info': operator_info,
                        'thread_count': thread_count,
                        'per_iter_count': metrics.get('per_iter_count', 0)
                    }
                
                # Add data for this matrix size
                thread_data[thread_count][function_name][key]['sizes'][matrix_size] = {
                    'memory_footprint_bytes': metrics['memory_footprint_bytes'],
                    'memory_footprint_str': metrics['memory_footprint_str'],
                    'execution_time_avg': metrics['execution_time_avg'],
                    'execution_time_min': metrics['execution_time_min'],
                    'execution_time_max': metrics['execution_time_max'],
                    'models': []
                }
                
                # Add model data
                for model in metrics.get('models', []):
                    model_data = {
                        'model_name': model.get('model_name', ''),
                        'aggregator': model.get('aggregator', ''),
                        'cost': model.get('cost', 0.0),
                        'footprint': model.get('footprint', '0 B')
                    }
                    thread_data[thread_count][function_name][key]['sizes'][matrix_size]['models'].append(model_data)
    
    return thread_data

def calculate_cache_thresholds(data_points):
    """
    Calculate matrix sizes where memory usage hits cache thresholds.
    Returns a dictionary of cache level -> matrix size thresholds.
    """
    thresholds = {}
    
    # Check if we have enough data points with non-zero memory footprint
    valid_points = [(d['matrix_size'], d['memory_footprint_bytes']) 
                   for d in data_points if d['memory_footprint_bytes'] > 0]
    
    if len(valid_points) < 2:
        return thresholds
    
    # Sort by matrix size
    valid_points.sort(key=lambda x: x[0])
    
    # Calculate average memory per element (relative memory usage)
    rel_mems = []
    for size, mem in valid_points:
        rel_mem = mem / size
        rel_mems.append(rel_mem)
    
    # Use median relative memory to avoid outliers
    rel_mem = np.median(rel_mems)
    
    if rel_mem <= 0:
        return thresholds
    
    # Calculate thresholds for each cache level
    for cache_name, cache_size in CACHE_SIZES.items():
        threshold_size = int(cache_size / rel_mem)
        thresholds[cache_name] = threshold_size
    
    return thresholds

def plot_results(function_data, output_dir="plots"):
    """Create plots for each function's performance metrics with different argument types in subplots."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define markers to use
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    
    # Create a non-red color palette for time lines
    colors = plt.cm.tab10.colors
    non_red_colors = [c for c in colors if c[0] < 0.7 or c[1] > 0.3]
    
    # Create a red color palette for cost lines
    red_colors = [
        '#FF0000',  # bright red
        '#CC0000',  # darker red
        '#990000',  # very dark red
        '#FF3333',  # lighter red
        '#FF6666',  # pale red
        '#800000',  # maroon
        '#A52A2A',  # brown
        '#B22222',  # firebrick
        '#DC143C',  # crimson
        '#CD5C5C',  # indian red
        '#FF4500',  # orange red
        '#FF8C00',  # dark orange
        '#C71585'   # medium violet red
    ]
    
    # For each function name, create a figure with subplots
    for function_name, args_data in function_data.items():
        # Group function variations by argument type signature and operator
        arg_categories = {}
        
        for args_key, data in args_data.items():
            # Extract signature information (operator type, args types)
            operator_info = data['operator_info']
            
            # Determine the argument pattern type for better categorization
            arg_pattern = "default"
            
            # Check for Vector vs scalar arguments patterns
            if "Vector" in args_key and "scalar" in args_key:
                arg_pattern = "vector_scalar"
            elif "Vector" in args_key and "Vector" in args_key[args_key.find("Vector")+6:]:
                arg_pattern = "vector_vector"
            elif "Vector" in args_key:
                arg_pattern = "vector_only"
            elif "scalar" in args_key:
                arg_pattern = "scalar_only"
            
            # More specific matching for function types from the args_key
            if function_name == "foldr" or function_name == "foldl":
                if "double" in args_key and "Vector" in args_key:
                    arg_pattern = "scalar_vector"
                elif "Vector" in args_key and "double" in args_key:
                    arg_pattern = "vector_scalar"
                elif "Vector" in args_key and "Vector" in args_key[args_key.find("Vector")+6:]:
                    arg_pattern = "vector_vector"
                    
            # Check if there's a Monoid vs specific operator
            if "Monoid" in args_key:
                if operator_info:
                    arg_pattern += "_op_" + operator_info
                else:
                    arg_pattern += "_monoid"
            elif operator_info:
                arg_pattern += "_" + operator_info
            
            # Create a category key combining arg pattern and operator info
            category_key = arg_pattern
            
            # Initialize this category if not seen before
            if category_key not in arg_categories:
                arg_categories[category_key] = []
            
            # Add this argument variation to the category
            arg_categories[category_key].append(args_key)
        
        # Determine subplot grid dimensions
        num_categories = len(arg_categories)
        if num_categories == 0:
            continue
            
        # Adjust figure size based on number of categories
        if num_categories == 1:
            # Single plot - use a more appropriate size
            fig = plt.figure(figsize=(10, 8))
            grid_cols = 1
            grid_rows = 1
        else:
            # Calculate grid dimensions (try to make it somewhat square)
            grid_cols = min(3, num_categories)  # Max 3 columns
            grid_rows = (num_categories + grid_cols - 1) // grid_cols
            fig = plt.figure(figsize=(6*grid_cols, 5*grid_rows))
        
        # Process each category in a separate subplot
        for idx, (category_key, args_keys) in enumerate(arg_categories.items()):
            # Create subplot - ONLY ONE Y-AXIS NOW
            ax = fig.add_subplot(grid_rows, grid_cols, idx+1)
            
            # Add subplot title with better formatting
            subplot_title = f"{function_name}"
            if category_key != "default":
                # Format the category key for display
                display_category = category_key.replace("_", " ").replace("vector", "Vector")
                display_category = display_category.replace("scalar", "Scalar").replace("monoid", "Monoid")
                subplot_title += f" ({display_category})"
            ax.set_title(subplot_title)
            
            # Keep track of lines for the legend
            all_lines = []
            all_labels = []
            
            # Process each argument variation in this category
            for i, args_key in enumerate(args_keys):
                data = args_data[args_key]
                
                # Extract information about this function variation
                thread_count = data['thread_count']
                per_iter_count = data['per_iter_count']
                
                # Create a simplified display name with just the per-iteration count
                display_name = ""
                
                # Add per-iteration count if available
                if per_iter_count is not None and per_iter_count > 0:
                    display_name = f"[{per_iter_count}/iter]"
                
                # Prepare data for plotting (X = memory footprint, Y = time/cost)
                footprints = []
                exec_times_avg = []
                exec_times_min = []
                exec_times_max = []
                
                # Store model data separately for each model/aggregator pair
                model_data = defaultdict(lambda: {'footprints': [], 'costs': []})
                
                # Extract data points from different matrix sizes
                for size, size_data in sorted(data['sizes'].items()):
                    # Only include sizes with valid memory footprint
                    if size_data['memory_footprint_bytes'] > 0:
                        # Convert to KB for better scale
                        memory_kb = size_data['memory_footprint_bytes'] / 1024.0
                        
                        # Add execution time data point
                        footprints.append(memory_kb)
                        exec_times_avg.append(size_data['execution_time_avg'])
                        exec_times_min.append(size_data['execution_time_min'])
                        exec_times_max.append(size_data['execution_time_max'])
                        
                        # Process model data
                        for model in size_data['models']:
                            model_key = (model['model_name'], model['aggregator'])
                            model_data[model_key]['footprints'].append(memory_kb)
                            model_data[model_key]['costs'].append(model['cost'])
                
                # Skip if no valid data points
                if not footprints:
                    continue
                    
                # Choose a color and marker for this function variation
                time_color = non_red_colors[i % len(non_red_colors)]
                marker = markers[i % len(markers)]
                
                # Plot execution time with this marker and color
                legend_name = "Execution Time"
                if display_name:
                    legend_name += f" {display_name}"
                
                time_line, = ax.plot(footprints, exec_times_avg, '-', marker=marker, color=time_color, 
                               label=legend_name)
                ax.fill_between(footprints, exec_times_min, exec_times_max, color=time_color, alpha=0.2)
                
                # Add time line to legend
                all_lines.append(time_line)
                all_labels.append(legend_name)
                
                # Plot a cost line for each model/aggregator pair ON THE SAME AXIS
                for j, (model_key, m_data) in enumerate(model_data.items()):
                    model_name, aggregator = model_key
                    
                    # Skip if no valid data
                    if not m_data['footprints']:
                        continue
                        
                    # Choose a red color variant for this model
                    cost_color = red_colors[j % len(red_colors)]
                    
                    # Create simplified model display name
                    if aggregator and aggregator != "default":
                        model_display = f"{model_name} {aggregator}"
                    else:
                        model_display = model_name
                    
                    # Plot cost line on the same axis
                    cost_line, = ax.plot(m_data['footprints'], m_data['costs'], '--', marker=marker, color=cost_color, 
                                   label=f"Cost Model: {model_display}")
                    
                    # Add cost line to legend
                    all_lines.append(cost_line)
                    all_labels.append(f"Cost Model: {model_display}")
            
            # Add vertical lines for cache sizes
            for cache_name, cache_size in CACHE_SIZES.items():
                # Convert cache size to KB for consistent x-axis
                cache_kb = cache_size / 1024.0
                ax.axvline(x=cache_kb, color='gray', linestyle='--', alpha=0.7)
                
                # Add cache size annotation near x-axis instead of at the top
                if cache_size < 1024 * 1024:  # Less than 1 MB
                    label = f"{cache_name} ({cache_kb:.0f} KB)"
                else:  # MB or larger
                    label = f"{cache_name} ({cache_kb/1024:.0f} MB)"
                
                # Position the cache size labels at the bottom near the x-axis
                ax.annotate(label, 
                           (cache_kb, ax.get_ylim()[0] * 1.1),  # Position near bottom
                           xytext=(0, 10),  # Offset text slightly above the x-axis
                           textcoords="offset points",
                           ha='center',  # Center horizontally on the line
                           va='bottom',
                           fontsize=8,
                           rotation=90,
                           color='black')
            
            # Configure axes - single y-axis for both time and cost
            ax.set_xlabel('Memory Footprint (KB)')
            ax.set_ylabel('Time / Cost (seconds)', color='black')  # Updated label for combined axis
            
            # Set to log scale
            ax.set_xscale('log', basex=2)
            ax.set_yscale('log')
            
            # Grid
            ax.grid(True, which="both", ls="--", alpha=0.3)
            
            # Add legend for this subplot - always at the upper left
            if all_lines:  # Only add legend if we have lines to show
                if num_categories == 1:
                    # For single plot, use larger font
                    ax.legend(all_lines, all_labels, loc='upper left', fontsize=9)
                else:
                    # For multi-plot, keep legend compact
                    ax.legend(all_lines, all_labels, loc='upper left', fontsize=7)
        
        # Add a main title for the whole figure
        plt.suptitle(f'{function_name} - Performance vs. Memory Footprint', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
        
        # Save figure with high resolution
        plt.savefig(os.path.join(output_dir, f'{function_name}_performance.png'), dpi=150)
        plt.close()

def plot_iteration_percentages(thread_data, output_dir="plots"):
    """
    Create a stacked bar chart showing what percentage of the Solver_iteration time 
    is taken by each function that runs per iteration, for all matrix sizes.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper function to determine text color based on background color
    def get_text_color(bg_color):
        # Convert color to RGB if it's not already
        if isinstance(bg_color, str):
            bg_color = mcolors.to_rgb(bg_color)
        
        # Calculate perceived brightness (luminance)
        # Using the formula: 0.299*R + 0.587*G + 0.114*B
        luminance = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])
        
        # Use black text on bright backgrounds, white text on dark backgrounds
        return 'black' if luminance > 0.6 else 'white'
    
    # Helper function to format size with appropriate units
    def format_size(size_kb):
        if size_kb >= 1024 * 1024:  # >= 1 GB
            return f"{size_kb / (1024 * 1024):.1f} GB"
        elif size_kb >= 1024:  # >= 1 MB
            return f"{size_kb / 1024:.1f} MB"
        else:
            return f"{size_kb:.0f} KB"
    
    # Helper function to format matrix dimensions with K, M, B units
    def format_dimension(dim):
        if dim >= 1024 * 1024 * 1024:  # >= 1B (1,073,741,824)
            return f"{dim//(1024*1024*1024)}B"
        elif dim >= 1024 * 1024:  # >= 1M (1,048,576)
            return f"{dim//(1024*1024)}M"
        elif dim >= 1024:  # >= 1K
            return f"{dim//1024}K"
        else:
            return str(dim)
    
    # For each thread count
    for thread_count, function_data in thread_data.items():
        # Check if we have Solver_iteration data
        if 'Solver_iteration' not in function_data:
            print(f"Warning: No Solver_iteration data found for {thread_count} threads, skipping percentage plot")
            continue
        
        # Get Solver_iteration data
        solver_data = function_data['Solver_iteration']
        solver_args = next(iter(solver_data))
        solver_sizes = solver_data[solver_args]['sizes']
        
        # Create a dictionary to store data for all matrix sizes
        all_size_data = {}
        
        # Process each matrix size
        for matrix_size, solver_size_data in sorted(solver_sizes.items()):
            solver_time = solver_size_data['execution_time_avg']
            
            if solver_time <= 0:
                print(f"Warning: Invalid Solver_iteration time for size {matrix_size}, skipping")
                continue
            
            # Initialize storage for this matrix size
            size_functions = []
            size_percentages = {}
            size_footprint = solver_size_data['memory_footprint_bytes'] / 1024.0  # KB
            
            # Find all functions with per_iter_count > 0
            for function_name, args_data in function_data.items():
                if function_name == 'Solver_iteration':
                    continue
                    
                for args_key, data in args_data.items():
                    per_iter_count = data.get('per_iter_count', 0)
                    
                    if per_iter_count and per_iter_count > 0:
                        # Get the execution time for this matrix size
                        if matrix_size in data['sizes']:
                            function_time = data['sizes'][matrix_size]['execution_time_avg']
                            
                            # Calculate percentage of solver time
                            percentage = (function_time * per_iter_count / solver_time) * 100.0
                            
                            # Skip functions with 0% contribution
                            if percentage <= 0:
                                continue
                            
                            # Get operation type for display
                            operator_info = data['operator_info']
                            display_name = function_name
                            if operator_info:
                                display_name = f"{function_name} ({operator_info})"
                            
                            # Add details to simplify display in multi-function cases
                            if args_key and 'Vector' in args_key and 'scalar' in args_key:
                                display_name += " (Vec+scalar)"
                            elif args_key and 'Vector' in args_key and 'Vector' in args_key[args_key.find('Vector')+6:]:
                                display_name += " (Vec+Vec)"
                            
                            # Add to function list if not already there
                            if display_name not in size_functions:
                                size_functions.append(display_name)
                            
                            # Store percentage data
                            size_percentages[display_name] = {
                                'percentage': percentage,
                                'time': function_time,
                                'per_iter': per_iter_count
                            }
            
            # Store data for this size
            all_size_data[matrix_size] = {
                'footprint': size_footprint,
                'functions': size_functions,
                'percentages': size_percentages,
                'solver_time': solver_time
            }
        
        # If we have no valid data, skip
        if not all_size_data:
            print(f"No valid data for thread count {thread_count}, skipping")
            continue
        
        # Get unique set of all functions across all sizes with non-zero percentages
        all_functions = set()
        for size_data in all_size_data.values():
            for func_name, func_data in size_data['percentages'].items():
                if func_data['percentage'] > 0:  # Only include functions with non-zero percentage
                    all_functions.add(func_name)
        
        # Sort functions by their maximum percentage across all sizes
        function_max_percentages = {}
        for func in all_functions:
            max_pct = 0
            for size_data in all_size_data.values():
                if func in size_data['percentages']:
                    max_pct = max(max_pct, size_data['percentages'][func]['percentage'])
            function_max_percentages[func] = max_pct
        
        # Sort functions by max percentage (descending)
        sorted_functions = sorted(all_functions, key=lambda f: function_max_percentages[f], reverse=True)
        
        # Create consistent colors for each function
        color_map = {}
        colormap = plt.cm.viridis
        for i, func in enumerate(sorted_functions):
            color_map[func] = colormap(i / max(1, len(sorted_functions) - 1))
        
        # Create figure with more width for the actual plot
        plt.figure(figsize=(16, 8))  # Reduced height slightly
        
        # Prepare data for plotting
        matrix_sizes = sorted(all_size_data.keys())
        x_labels = []
        x_positions = []
        bottom_values = np.zeros(len(matrix_sizes))
        
        # Create x-axis labels with formatted dimensions and memory size
        for i, size in enumerate(matrix_sizes):
            footprint = all_size_data[size]['footprint']
            formatted_dim = f"{format_dimension(size)}×{format_dimension(size)}"
            formatted_footprint = format_size(footprint)
            x_labels.append(f"{formatted_dim}\n({formatted_footprint})")
            x_positions.append(i)
        
        # Plot stacked bars for each function
        handles = []  # For legend
        labels = []   # For legend
        
        # Add "Total (μs)" annotation near the top-right of the y-axis
        plt.annotate("Total (μs)", xy=(0, 1.05), xycoords=('axes fraction', 'axes fraction'),
                     ha='left', va='center', fontsize=9)
        
        for func in sorted_functions:
            values = []
            for size in matrix_sizes:
                if func in all_size_data[size]['percentages']:
                    values.append(all_size_data[size]['percentages'][func]['percentage'])
                else:
                    values.append(0)
            
            # Skip function if all values are zero
            if all(v == 0 for v in values):
                continue
                
            # Plot this function's bars
            bar = plt.bar(x_positions, values, bottom=bottom_values, label=func, color=color_map[func])
            
            # Store for legend
            handles.append(bar)
            labels.append(func)
            
            # Add percentage labels inside the bars if they're large enough
            for i, v in enumerate(values):
                if v >= 5.0:  # Only show label if percentage is at least 5%
                    # Get data for annotation
                    size = matrix_sizes[i]
                    func_data = all_size_data[size]['percentages'].get(func, {})
                    time_us = func_data.get('time', 0) * 1e6
                    per_iter = func_data.get('per_iter', 0)
                    
                    # Position label in the middle of this function's segment
                    y_pos = bottom_values[i] + v/2
                    
                    # Get the color of this segment
                    segment_color = color_map[func]
                    
                    # Determine optimal text color for this background
                    text_color = get_text_color(segment_color)
                    
                    # Add text with percentage and time - using dynamic text color
                    plt.text(x_positions[i], y_pos, f"{v:.1f}%\n({time_us:.1f}μs×{per_iter})", 
                             ha='center', va='center', fontsize=8, 
                             color=text_color)  # Dynamic text color
            
            # Update bottom values for next function
            bottom_values = bottom_values + values
        
        # Add execution time above each bar (just the value, no "Total: " or "μs")
        for i, size in enumerate(matrix_sizes):
            solver_time_us = all_size_data[size]['solver_time'] * 1e6
            plt.text(x_positions[i], 105, f"{solver_time_us:.1f}", 
                     ha='center', va='bottom', fontsize=9)
        
        # Set up the plot
        plt.ylabel('Percentage of Solver Iteration Time (%)')
        plt.title(f'Function Time Distribution per Iteration ({thread_count} threads)', fontsize=14, pad=15)  # Added padding
        plt.xticks(x_positions, x_labels)
        plt.ylim(0, 108)  # Increased slightly for more space at top
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Position the legend below the plot with less space
        legend_cols = min(5, len(sorted_functions))  # Max 5 columns, adjust as needed
        
        # Add the legend below the plot with reduced spacing
        plt.legend(handles=handles, labels=labels, 
                  loc='upper center', bbox_to_anchor=(0.5, -0.08),  # Reduced space
                  ncol=legend_cols, fontsize=10)
        
        # Tight layout with room for legend but reduced bottom margin
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Reduced bottom margin
        
        # Save the plot with higher resolution but smaller margins
        plt.savefig(os.path.join(output_dir, f'iteration_percentages_{thread_count}t.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

def plot_cost_percentages(thread_data, output_dir="plots"):
    """
    Create a stacked bar chart showing what percentage of the total predicted cost
    is taken by each function that runs per iteration, for all matrix sizes.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper function to determine text color based on background color
    def get_text_color(bg_color):
        # Convert color to RGB if it's not already
        if isinstance(bg_color, str):
            bg_color = mcolors.to_rgb(bg_color)
        
        # Calculate perceived brightness (luminance)
        luminance = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])
        
        # Use black text on bright backgrounds, white text on dark backgrounds
        return 'black' if luminance > 0.6 else 'white'
    
    # Helper function to format size with appropriate units
    def format_size(size_kb):
        if size_kb >= 1024 * 1024:  # >= 1 GB
            return f"{size_kb / (1024 * 1024):.1f} GB"
        elif size_kb >= 1024:  # >= 1 MB
            return f"{size_kb / 1024:.1f} MB"
        else:
            return f"{size_kb:.0f} KB"
    
    # Helper function to format matrix dimensions with K, M, B units
    def format_dimension(dim):
        if dim >= 1024 * 1024 * 1024:  # >= 1B (1,073,741,824)
            return f"{dim//(1024*1024*1024)}B"
        elif dim >= 1024 * 1024:  # >= 1M (1,048,576)
            return f"{dim//(1024*1024)}M"
        elif dim >= 1024:  # >= 1K
            return f"{dim//1024}K"
        else:
            return str(dim)
    
    # For each thread count
    for thread_count, function_data in thread_data.items():
        # Create a dictionary to store data for all matrix sizes
        all_size_data = {}
        
        # Process each matrix size
        matrix_sizes = set()
        for function_name, args_data in function_data.items():
            for args_key, data in args_data.items():
                per_iter_count = data.get('per_iter_count', 0)
                if per_iter_count and per_iter_count > 0:
                    for size in data['sizes']:
                        matrix_sizes.add(size)
        
        matrix_sizes = sorted(matrix_sizes)
        
        # For each matrix size, calculate total cost and percentage breakdown
        for matrix_size in matrix_sizes:
            # Initialize storage for this matrix size
            size_functions = []
            size_percentages = {}
            size_footprint = 0
            total_cost = 0
            
            # First pass: calculate total cost for this matrix size
            for function_name, args_data in function_data.items():
                for args_key, data in args_data.items():
                    per_iter_count = data.get('per_iter_count', 0)
                    
                    if per_iter_count and per_iter_count > 0:
                        # Check if we have cost data for this matrix size
                        if matrix_size in data['sizes']:
                            size_data = data['sizes'][matrix_size]
                            
                            # Update memory footprint if available
                            if size_footprint == 0 and size_data['memory_footprint_bytes'] > 0:
                                size_footprint = size_data['memory_footprint_bytes'] / 1024.0  # KB
                            
                            # Get cost from first available model (assuming consistent models)
                            for model in size_data['models']:
                                if model.get('cost') is not None:
                                    # Add to total cost (cost × per_iter_count)
                                    function_cost = model.get('cost', 0.0) * per_iter_count
                                    total_cost += function_cost
                                    break
            
            # Skip if no cost data found
            if total_cost <= 0:
                print(f"Warning: No valid cost data for size {matrix_size}, skipping")
                continue
            
            # Second pass: calculate percentages
            for function_name, args_data in function_data.items():
                for args_key, data in args_data.items():
                    per_iter_count = data.get('per_iter_count', 0)
                    
                    if per_iter_count and per_iter_count > 0:
                        # Check if we have cost data for this matrix size
                        if matrix_size in data['sizes']:
                            size_data = data['sizes'][matrix_size]
                            
                            # Get cost from first available model
                            model_cost = None
                            model_name = None
                            for model in size_data['models']:
                                if model.get('cost') is not None:
                                    model_cost = model.get('cost', 0.0)
                                    model_name = model.get('model_name', '')
                                    break
                            
                            if model_cost is not None:
                                # Calculate percentage of total cost
                                function_cost = model_cost * per_iter_count
                                percentage = (function_cost / total_cost) * 100.0
                                
                                # Skip functions with negligible contribution
                                if percentage <= 0.01:
                                    continue
                                
                                # Get operation type for display
                                operator_info = data['operator_info']
                                display_name = function_name
                                if operator_info:
                                    display_name = f"{function_name} ({operator_info})"
                                
                                # Add details to simplify display in multi-function cases
                                if args_key and 'Vector' in args_key and 'scalar' in args_key:
                                    display_name += " (Vec+scalar)"
                                elif args_key and 'Vector' in args_key and 'Vector' in args_key[args_key.find('Vector')+6:]:
                                    display_name += " (Vec+Vec)"
                                
                                # Add to function list if not already there
                                if display_name not in size_functions:
                                    size_functions.append(display_name)
                                
                                # Store percentage data
                                size_percentages[display_name] = {
                                    'percentage': percentage,
                                    'cost': model_cost,
                                    'total_cost': function_cost,
                                    'per_iter': per_iter_count,
                                    'model': model_name
                                }
            
            # Store data for this size
            all_size_data[matrix_size] = {
                'footprint': size_footprint,
                'functions': size_functions,
                'percentages': size_percentages,
                'total_cost': total_cost
            }
        
        # If we have no valid data, skip
        if not all_size_data:
            print(f"No valid cost data for thread count {thread_count}, skipping")
            continue
        
        # Get unique set of all functions across all sizes with non-zero percentages
        all_functions = set()
        for size_data in all_size_data.values():
            for func_name, func_data in size_data['percentages'].items():
                if func_data['percentage'] > 0:  # Only include functions with non-zero percentage
                    all_functions.add(func_name)
        
        # Sort functions by their maximum percentage across all sizes
        function_max_percentages = {}
        for func in all_functions:
            max_pct = 0
            for size_data in all_size_data.values():
                if func in size_data['percentages']:
                    max_pct = max(max_pct, size_data['percentages'][func]['percentage'])
            function_max_percentages[func] = max_pct
        
        # Sort functions by max percentage (descending)
        sorted_functions = sorted(all_functions, key=lambda f: function_max_percentages[f], reverse=True)
        
        # Create consistent colors for each function
        color_map = {}
        colormap = plt.cm.viridis
        for i, func in enumerate(sorted_functions):
            color_map[func] = colormap(i / max(1, len(sorted_functions) - 1))
        
        # Create figure with more width for the actual plot
        plt.figure(figsize=(16, 8))
        
        # Prepare data for plotting
        matrix_sizes = sorted(all_size_data.keys())
        x_labels = []
        x_positions = []
        bottom_values = np.zeros(len(matrix_sizes))
        
        # Create x-axis labels with formatted dimensions and memory size
        for i, size in enumerate(matrix_sizes):
            footprint = all_size_data[size]['footprint']
            formatted_dim = f"{format_dimension(size)}×{format_dimension(size)}"
            formatted_footprint = format_size(footprint)
            x_labels.append(f"{formatted_dim}\n({formatted_footprint})")
            x_positions.append(i)
        
        # Plot stacked bars for each function
        handles = []  # For legend
        labels = []   # For legend
        
        # Add "Total Cost" annotation near the top-left of the y-axis
        plt.annotate("Total Cost", xy=(0, 1.05), xycoords=('axes fraction', 'axes fraction'),
                     ha='left', va='center', fontsize=9)
        
        for func in sorted_functions:
            values = []
            for size in matrix_sizes:
                if func in all_size_data[size]['percentages']:
                    values.append(all_size_data[size]['percentages'][func]['percentage'])
                else:
                    values.append(0)
            
            # Skip function if all values are zero
            if all(v == 0 for v in values):
                continue
                
            # Plot this function's bars
            bar = plt.bar(x_positions, values, bottom=bottom_values, label=func, color=color_map[func])
            
            # Store for legend
            handles.append(bar)
            labels.append(func)
            
            # Add percentage labels inside the bars if they're large enough
            for i, v in enumerate(values):
                if v >= 5.0:  # Only show label if percentage is at least 5%
                    # Get data for annotation
                    size = matrix_sizes[i]
                    func_data = all_size_data[size]['percentages'].get(func, {})
                    cost_value = func_data.get('cost', 0)
                    per_iter = func_data.get('per_iter', 0)
                    model_name = func_data.get('model', '')
                    
                    # Position label in the middle of this function's segment
                    y_pos = bottom_values[i] + v/2
                    
                    # Get the color of this segment
                    segment_color = color_map[func]
                    
                    # Determine optimal text color for this background
                    text_color = get_text_color(segment_color)
                    
                    # Add text with percentage and cost
                    plt.text(x_positions[i], y_pos, f"{v:.1f}%\n({cost_value:.2e}×{per_iter})", 
                             ha='center', va='center', fontsize=8, 
                             color=text_color)  # Dynamic text color
            
            # Update bottom values for next function
            bottom_values = bottom_values + values
        
        # Add total cost above each bar
        for i, size in enumerate(matrix_sizes):
            total = all_size_data[size]['total_cost']
            plt.text(x_positions[i], 105, f"{total:.2e}", 
                     ha='center', va='bottom', fontsize=9)
        
        # Set up the plot
        plt.ylabel('Percentage of Total Predicted Cost (%)')
        plt.title(f'Function Cost Distribution per Iteration ({thread_count} threads)', fontsize=14, pad=15)
        plt.xticks(x_positions, x_labels)
        plt.ylim(0, 108)  # Leave room for total cost label
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Position the legend below the plot
        legend_cols = min(5, len(sorted_functions))
        plt.legend(handles=handles, labels=labels, 
                  loc='upper center', bbox_to_anchor=(0.5, -0.08),
                  ncol=legend_cols, fontsize=10)
        
        # Tight layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'cost_percentages_{thread_count}t.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
# Update main function to include the new plot
def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Plot performance data from analysis files.')
    parser.add_argument('--results-dir', default='results', 
                        help='Directory containing the analysis files (default: results)')
    parser.add_argument('--threads', type=int, default=None,
                        help='Only plot results for this thread count (default: all)')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    plots_base_dir = os.path.join(results_dir, 'plots')
    thread_data = collect_all_results(results_dir)
    
    # If --threads is specified, only plot for that thread count
    thread_counts = [args.threads] if args.threads is not None else sorted(thread_data.keys())
    
    for thread_count in thread_counts:
        if thread_count not in thread_data:
            print(f"No data found for {thread_count} threads, skipping.")
            continue
        function_data = thread_data[thread_count]
        thread_plots_dir = os.path.join(plots_base_dir, f't{thread_count}')
        os.makedirs(thread_plots_dir, exist_ok=True)
        function_count = len(function_data)
        print(f"Found data for {function_count} distinct function variations with {thread_count} threads")
        print(f"Generating performance plots in: {thread_plots_dir}")
        plot_results(function_data, thread_plots_dir)
        print(f"Generating time percentage plots in: {thread_plots_dir}")
        plot_iteration_percentages({thread_count: function_data}, thread_plots_dir)
        print(f"Generating cost percentage plots in: {thread_plots_dir}")
        plot_cost_percentages({thread_count: function_data}, thread_plots_dir)
        print(f"Plotting complete for {thread_count} threads!")
    
    print("All plotting tasks completed!")

if __name__ == "__main__":
    main()