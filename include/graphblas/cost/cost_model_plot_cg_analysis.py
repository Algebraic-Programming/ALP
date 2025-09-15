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

    # First try the _threads-X pattern
    match = re.search(r'banded_diag_(\d+)x(\d+)_band_(\d+)_threads-(\d+)_analysis\.log', filename)
    if match:
        print(f"extract_info_from_filename({filename}): size = {match.group(1)}, threads = {match.group(4)}")
        return int(match.group(1)), int(match.group(4))
    
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
    Create a simplified representation of args that distinguishes different operators
    but removes specific size/nnz values.
    """
    # Extract data type
    dtype_match = re.search(r'<(\w+)>', args)
    dtype = dtype_match.group(1) if dtype_match else ""
    
    # Extract operator information
    operator_info = extract_operator_info(args)
    
    if dtype and operator_info:
        return f"{dtype}_{operator_info}"
    elif dtype:
        return dtype
    else:
        # Fallback to a simplified version of the full args
        simplified = re.sub(r'size=\d+', 'size=X', args)
        simplified = re.sub(r'rows=\d+,cols=\d+', 'rows=X,cols=X', simplified)
        simplified = re.sub(r'nnz=\d+', 'nnz=X', simplified)
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
    
    # Get matrix size and thread count from filename
    matrix_size, thread_count = extract_info_from_filename(os.path.basename(filepath))
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Find function analysis sections
            if line.startswith('Analysis for function'):
                match = re.search(r"Analysis for function '(\w+)':", line)
                if match:
                    current_function = match.group(1)
                    result[current_function] = defaultdict(dict)
            
            # Find argument types
            elif line.startswith('Argument types:') and current_function:
                current_args = line.replace('Argument types:', '').strip()
                # Initialize metrics for this function+args combination
                result[current_function][current_args] = {
                    'count': 0,
                    'cost': 0.0,
                    'execution_time_min': 0.0,
                    'execution_time_max': 0.0,
                    'execution_time_avg': 0.0,
                    'execution_time_stddev': 0.0,
                    'matrix_size': matrix_size,
                    'thread_count': thread_count,
                    'simplified_args': simplify_args(current_args),
                    'operator_info': extract_operator_info(current_args),
                    'memory_footprint_bytes': 0
                }
            
            # Find memory footprint
            elif line.startswith('Memory footprint:') and current_function and current_args:
                footprint_str = line.replace('Memory footprint:', '').strip()
                memory_bytes = parse_memory_footprint(footprint_str)
                result[current_function][current_args]['memory_footprint_bytes'] = memory_bytes
                result[current_function][current_args]['memory_footprint_str'] = footprint_str
            
            # Find predicted cost
            elif line.startswith('Predicted cost:') and current_function and current_args:
                cost = float(line.replace('Predicted cost:', '').strip())
                result[current_function][current_args]['cost'] = cost
            
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
    
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return {}  # Return empty dict on error
        
    return result

def collect_all_results(results_dir):
    """Collect results from all analysis files in the directory."""
    all_files = glob.glob(os.path.join(results_dir, '*_analysis.log'))
    
    # FIX: Use simple defaultdict(list) instead of nested defaultdict
    function_data = defaultdict(list)
    
    for filepath in all_files:
        matrix_size, thread_count = extract_info_from_filename(os.path.basename(filepath))
        
        if matrix_size is None:
            continue
            
        file_results = parse_analysis_file(filepath)
        
        # Skip if file_results is empty
        if not file_results:
            print(f"Warning: No valid data found in {filepath}")
            continue
            
        for function_name, args_data in file_results.items():
            for args, metrics in args_data.items():
                # Create a key that includes function name and operator info
                # This ensures different operators get different lines
                operator_info = metrics['operator_info']
                key = f"{function_name}_{operator_info}" if operator_info else function_name
                
                # For multiple thread counts, add thread count to the key
                if thread_count > 1:
                    key = f"{key}_{thread_count}t"
                
                # Add this data point
                function_data[key].append({
                    'matrix_size': matrix_size,
                    'thread_count': thread_count,
                    'count': metrics['count'],
                    'cost': metrics['cost'],
                    'execution_time_avg': metrics['execution_time_avg'],
                    'execution_time_min': metrics['execution_time_min'],
                    'execution_time_max': metrics['execution_time_max'],
                    'execution_time_stddev': metrics['execution_time_stddev'],
                    'simplified_args': metrics['simplified_args'],
                    'operator_info': metrics['operator_info'],
                    'memory_footprint_bytes': metrics['memory_footprint_bytes'],
                    'memory_footprint_str': metrics.get('memory_footprint_str', '0 B')
                })
    
    return function_data

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
    """Create plots for each function's performance metrics."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group functions by their base name for plotting together
    base_functions = defaultdict(list)
    for func_key in function_data:
        # Extract base function name (before the underscore)
        base_name = func_key.split('_')[0]
        base_functions[base_name].append(func_key)
    
    # Define markers to use
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    
    # Plot each base function
    for base_name, func_keys in base_functions.items():
        # Create a figure with a single plot and twin y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()  # Create a secondary y-axis
        
        # Keep track of lines for the legend
        all_lines = []
        all_labels = []
        
        # Create a non-red color palette
        colors = plt.cm.tab10.colors
        non_red_colors = [c for c in colors if c[0] < 0.7 or c[1] > 0.3]
        
        # For each function/operator variation
        for i, func_key in enumerate(sorted(func_keys)):
            # Group data by matrix size
            size_data = defaultdict(list)
            for data_point in function_data[func_key]:
                size_data[data_point['matrix_size']].append(data_point)
            
            # Calculate averages for each size
            sizes = []
            costs = []
            exec_times_avg = []
            exec_times_min = []
            exec_times_max = []
            memory_footprints = []
            
            for size in sorted(size_data.keys()):
                data_points = size_data[size]
                sizes.append(size)
                costs.append(np.mean([d['cost'] for d in data_points]))
                exec_times_avg.append(np.mean([d['execution_time_avg'] for d in data_points]))
                exec_times_min.append(np.mean([d['execution_time_min'] for d in data_points]))
                exec_times_max.append(np.mean([d['execution_time_max'] for d in data_points]))
                memory_footprints.append(np.mean([d['memory_footprint_bytes'] for d in data_points]))
            
            # Get the simplified name for display
            if len(function_data[func_key]) > 0:
                operator_info = function_data[func_key][0]['operator_info']
                display_name = f"{base_name} ({operator_info})" if operator_info else base_name
            else:
                display_name = func_key
            
            # Choose a marker for this function variation
            marker = markers[i % len(markers)]
            
            # Choose a color for this function variation
            color = non_red_colors[i % len(non_red_colors)]
            
            # Plot execution time with this marker and color
            time_line, = ax1.plot(sizes, exec_times_avg, '-', marker=marker, color=color, 
                           label=f"Time: {display_name}")
            ax1.fill_between(sizes, exec_times_min, exec_times_max, color=color, alpha=0.2)
            
            # Plot cost with the same marker but in red and dashed
            cost_line, = ax2.plot(sizes, costs, '--', marker=marker, color='red', 
                           label=f"Cost: {display_name}")
            
            # Add to legend
            all_lines.extend([time_line, cost_line])
            all_labels.extend([f"Time: {display_name}", f"Cost: {display_name}"])
            
            # Calculate cache thresholds
            cache_thresholds = calculate_cache_thresholds(function_data[func_key])
            
            # Add cache threshold lines using the same color as the function
            # but with a dotted style and without adding to legend
            for cache_name, threshold in cache_thresholds.items():
                if threshold > 0:
                    # Add a dotted line with the same color as the function
                    ax1.axvline(x=threshold, color=color, linestyle=':', alpha=0.7, linewidth=1.5)
                    
                    # Add a small annotation
                    ax1.annotate(f"{cache_name}", 
                               (threshold, ax1.get_ylim()[1]*0.95),
                               xytext=(5, 0),
                               textcoords="offset points",
                               ha='left',
                               va='center',
                               fontsize=8,
                               rotation=90,
                               color=color)
        
        # Configure axes
        ax1.set_xlabel('Matrix Size (N)')
        ax1.set_ylabel('Execution Time (seconds)', color='blue')
        ax2.set_ylabel('Predicted Cost', color='red')
        
        # Set to log scale
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        
        # Grid and title
        ax1.grid(True, which="both", ls="--", alpha=0.3)
        plt.title(f'{base_name} - Performance vs. Matrix Size')
        
        # Add vertical lines at matrix sizes
        all_sizes = set()
        for func_key in func_keys:
            for data_point in function_data[func_key]:
                all_sizes.add(data_point['matrix_size'])
        
        for size in sorted(all_sizes):
            ax1.axvline(x=size, color='gray', linestyle=':', alpha=0.3)
            # Add size annotation at the bottom
            ax1.annotate(f"{size}", 
                       (size, ax1.get_ylim()[0]*1.1),
                       textcoords="offset points",
                       xytext=(0,5),
                       ha='center',
                       fontsize=8,
                       rotation=90)
        
        # Add legend for function lines only (cache lines excluded)
        plt.legend(all_lines, all_labels, loc='best')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_name}_performance.png'))
        plt.close()

def main():
    # Directory containing the analysis files
    results_dir = 'results'
    
    # Create plots directory
    plots_dir = 'results/plots'
    
    # Collect and process results
    print(f"Scanning results directory: {results_dir}")
    function_data = collect_all_results(results_dir)
    
    # Count the total number of functions found
    function_count = len(function_data)
    print(f"Found data for {function_count} distinct function variations")
    
    # Generate plots
    print(f"Generating plots in: {plots_dir}")
    plot_results(function_data, plots_dir)
    
    print("Plotting complete!")

if __name__ == "__main__":
    main()