#!/usr/bin/env python3
# filepath: /home/panastasiadis/ALP/include/graphblas/cost/benchmark_ploter.py
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

def extract_thread_count_from_path(filepath):
    """Extract thread count from the file path (e.g., from 't96' in the path)."""
    # Look for thread count pattern in the path (e.g., 't96', 't1', etc.)
    match = re.search(r'/t(\d+)/', filepath)
    if match:
        return int(match.group(1))
    
    # If no thread count found, print error and exit
    print(f"Error: Could not determine thread count from filepath: {filepath}")
    print("Expected patterns: /t<number>/ or _threads-<number>_ or _threads_<number>_ or _<number>t_")
    exit(1)

def parse_output_log_file(output_filepath):
    """Parse an output.log file to extract milliseconds per iteration data."""
    try:
        with open(output_filepath, 'r') as f:
            content = f.read()
        
        # Look for the "milliseconds per iteration" line
        match = re.search(r'milliseconds per iteration:\s+([\d.]+)', content)
        if match:
            milliseconds_per_iter = float(match.group(1))
            # Convert to seconds
            return milliseconds_per_iter / 1000.0
        else:
            print(f"Warning: Could not find 'milliseconds per iteration' in {output_filepath}")
            return None
    except Exception as e:
        print(f"Error reading output file {output_filepath}: {e}")
        return None

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

def parse_analysis_file(filepath):
    """Parse an analysis file to extract function metrics."""
    result = {}
    current_function = None
    current_args = None
    current_model_data = None
    
    # Get thread count from filepath
    thread_count = extract_thread_count_from_path(filepath)
    
    filename = os.path.basename(filepath)
    print(f"  Parsing file: {filename} (threads: {thread_count})")
    
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
                simplified_args = simplify_args(current_args)
                operator_info = extract_operator_info(current_args)
                
                # Initialize if not already present
                if current_args not in result[current_function]:
                    result[current_function][current_args] = {
                        'thread_count': thread_count,
                        'simplified_args': simplified_args,
                        'operator_info': operator_info,
                        'models': []
                    }
                
                # Get invocation count from next line if available
                if i + 1 < len(lines) and lines[i+1].strip().startswith('Invocation count:'):
                    count_line = lines[i+1].strip()
                    
                    # Extract both total count and per-iteration count
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
                model_name = line.replace('Model:', '').strip()
                current_model_data = {
                    'model_name': model_name,
                    'threads': None,
                    'aggregator': None,
                    'footprint': None,
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
                        
                        # Validate memory footprint consistency
                        if 'expected_footprint' not in result[current_function][current_args]:
                            result[current_function][current_args]['expected_footprint'] = footprint_str
                        elif result[current_function][current_args]['expected_footprint'] != footprint_str:
                            print(f"ERROR: Memory footprint mismatch for {current_function} {current_args}")
                            print(f"  Expected: {result[current_function][current_args]['expected_footprint']}")
                            print(f"  Found: {footprint_str}")
                            exit(1)
                            
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
    
    # Print summary of what was parsed (only for first file as example)
    if not hasattr(parse_analysis_file, '_example_shown'):
        print(f"    Example - Parsed {len(result)} functions:")
        for func_name, args_data in list(result.items())[:2]:  # Show first 2 functions as example
            print(f"      {func_name}: {len(args_data)} argument variations")
            for args, data in list(args_data.items())[:1]:  # Show first argument variation
                print(f"        {data['simplified_args']}: {len(data['models'])} models")
        if len(result) > 2:
            print(f"      ... and {len(result) - 2} more functions")
        parse_analysis_file._example_shown = True
    
    return result


def collect_all_results(results_dir, filegroup_name, threads, clearun_dir=None):
    """Collect results from all analysis files in the directory for a specific filegroup and thread count."""
    # Construct the path: results_dir/t{threads}/results/analysis/{filegroup_name}*
    analysis_dir = os.path.join(
        results_dir, f't{threads}', 'results', 'analysis', f'{filegroup_name}')

    if not os.path.exists(analysis_dir):
        print(f"Error: Analysis directory does not exist: {analysis_dir}")
        exit(1)

    # Find all files matching the filegroup pattern
    pattern = os.path.join(analysis_dir, f'*_analysis.log')
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"Error: No analysis files found matching pattern: {pattern}")
        exit(1)
    
    print(f"Found {len(all_files)} analysis files for filegroup '{filegroup_name}' with {threads} threads")
    
    # Group data by function -> simplified_args -> benchmark_file
    grouped_data = {}
    
    print(f"\nStarting to parse {len(all_files)} files...")
    for i, filepath in enumerate(all_files):
        if i == 0:
            print(f"\nParsing: {os.path.basename(filepath)}")
        else:
            print(f"Parsing: {os.path.basename(filepath)}")
        
        # Extract benchmark identifier from filename (remove _analysis.log suffix)
        benchmark_id = os.path.basename(filepath).replace('_analysis.log', '')
        
        file_results = parse_analysis_file(filepath)
        
        # Skip if file_results is empty
        if not file_results:
            print(f"Warning: No valid data found in {filepath}")
            continue
        
        # Try to get actual execution time from output.log file if clearun_dir is provided
        actual_execution_time = None
        if clearun_dir:
            output_filepath = os.path.join(clearun_dir, f't{threads}', 'outputs', filegroup_name, f'{benchmark_id}_output.log')
            if os.path.exists(output_filepath):
                actual_execution_time = parse_output_log_file(output_filepath)
                if actual_execution_time is not None:
                    print(f"  Found actual execution time: {actual_execution_time:.6f}s for {benchmark_id}")
            else:
                print(f"  Warning: Output file not found: {output_filepath}")
        
        for function_name, args_data in file_results.items():
            # Initialize function in data structure if needed
            if function_name not in grouped_data:
                grouped_data[function_name] = {}
            
            for args, metrics in args_data.items():
                # Create a key that removes size information
                simplified_args = simplify_args(args)
                
                # Initialize simplified args in data structure if needed
                if simplified_args not in grouped_data[function_name]:
                    grouped_data[function_name][simplified_args] = {
                        'operator_info': metrics.get('operator_info', ''),
                        'thread_count': metrics['thread_count'],
                        'per_iter_count': metrics.get('per_iter_count', 0),
                        'benchmarks': {}  # Group by benchmark file
                    }
                
                # Get memory footprint from the first model (should be same for all models)
                memory_footprint_str = metrics.get('expected_footprint', '0 B')
                memory_footprint_bytes = parse_memory_footprint(memory_footprint_str)
                
                # Store data for this benchmark
                grouped_data[function_name][simplified_args]['benchmarks'][benchmark_id] = {
                    'benchmark_id': benchmark_id,
                    'memory_footprint_str': memory_footprint_str,
                    'memory_footprint_bytes': memory_footprint_bytes,
                    'execution_time_avg': metrics['execution_time_avg'],
                    'execution_time_min': metrics['execution_time_min'],
                    'execution_time_max': metrics['execution_time_max'],
                    'execution_time_stddev': metrics['execution_time_stddev'],
                    'actual_execution_time': actual_execution_time,  # Add actual execution time if available
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
                    grouped_data[function_name][simplified_args]['benchmarks'][benchmark_id]['models'].append(model_data)
    
    # Print final grouping summary
    print(f"\n=== FINAL GROUPING SUMMARY ===")
    print(f"Total functions found: {len(grouped_data)}")
    for func_name, args_data in grouped_data.items():
        print(f"  {func_name}:")
        for simplified_args, data in args_data.items():
            benchmarks = sorted(data['benchmarks'].keys())
            print(f"    {simplified_args}: {len(benchmarks)} benchmarks {benchmarks}")
    
    # Print a single example of internal data structure
    print(f"\n=== EXAMPLE INTERNAL DATA STRUCTURE ===")
    if grouped_data:
        first_func = list(grouped_data.keys())[0]
        first_args = list(grouped_data[first_func].keys())[0]
        first_benchmark = list(grouped_data[first_func][first_args]['benchmarks'].keys())[0]
        
        example_data = grouped_data[first_func][first_args]['benchmarks'][first_benchmark]
        
        print(f"Function: {first_func}")
        print(f"Arguments: {first_args}")
        print(f"Benchmark: {example_data['benchmark_id']}")
        print(f"Memory footprint: {example_data['memory_footprint_str']}")
        print(f"Execution time (avg): {example_data['execution_time_avg']:.2e}s")
        print(f"Models found: {len(example_data['models'])}")
        for i, model in enumerate(example_data['models']):  # Show all models
            print(f"  Model {i+1}: {model['model_name']} - Cost: {model['cost']:.2e}, Footprint: {model['footprint']}")
    
    return grouped_data

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
                
                # Extract data points from different benchmarks, sorted by memory footprint
                sorted_benchmarks = sorted(data['benchmarks'].items(), 
                                         key=lambda x: x[1]['memory_footprint_bytes'])
                
                for benchmark_id, benchmark_data in sorted_benchmarks:
                    memory_footprint_bytes = benchmark_data['memory_footprint_bytes']
                    execution_time_avg = benchmark_data['execution_time_avg']
                    
                    # Skip functions with zero execution time
                    if execution_time_avg <= 0:
                        continue
                    
                    if memory_footprint_bytes > 0:
                        # Convert to KB for better scale
                        memory_kb = memory_footprint_bytes / 1024.0
                        
                        # Add execution time data point
                        footprints.append(memory_kb)
                        exec_times_avg.append(execution_time_avg)
                        exec_times_min.append(benchmark_data['execution_time_min'])
                        exec_times_max.append(benchmark_data['execution_time_max'])
                        
                        # Process model data
                        for model in benchmark_data['models']:
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
                
                # For Solver_iteration, add actual execution time as a green line if available
                if function_name == 'Solver_iteration':
                    actual_times = []
                    actual_footprints = []
                    
                    for benchmark_id, benchmark_data in sorted_benchmarks:
                        if benchmark_data.get('actual_execution_time') is not None:
                            actual_times.append(benchmark_data['actual_execution_time'])
                            actual_footprints.append(benchmark_data['memory_footprint_bytes'] / 1024.0)  # Convert to KB
                    
                    if actual_times:
                        actual_line, = ax.plot(actual_footprints, actual_times, '-', marker='o', color='green', 
                                             linewidth=2, markersize=6, label="Avg Isolated Execution Time [1/iter]")
                        all_lines.append(actual_line)
                        all_labels.append("Avg Isolated Execution Time [1/iter]")
            
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
            ax.set_xscale('log', base=2)
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
    is taken by each function that runs per iteration, for all benchmarks.
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
    
    # For each thread count
    for thread_count, function_data in thread_data.items():
        # Check if we have Solver_iteration data
        if 'Solver_iteration' not in function_data:
            print(f"Warning: No Solver_iteration data found for {thread_count} threads, skipping percentage plot")
            continue
        
        # Get Solver_iteration data
        solver_data = function_data['Solver_iteration']
        solver_args = next(iter(solver_data))
        solver_data_point = solver_data[solver_args]
        
        # Create a dictionary to store data for all benchmarks
        all_benchmark_data = {}
        
        # Process each benchmark for Solver_iteration
        for benchmark_id, solver_benchmark_data in sorted(solver_data_point['benchmarks'].items()):
            solver_time = solver_benchmark_data['execution_time_avg']
            
            if solver_time <= 0:
                print(f"Warning: Invalid Solver_iteration time for benchmark {benchmark_id}, skipping")
                continue
            
            # Initialize storage for this benchmark
            benchmark_functions = []
            benchmark_percentages = {}
            memory_footprint_bytes = solver_benchmark_data['memory_footprint_bytes']
            size_footprint = memory_footprint_bytes / 1024.0  # KB
            
            # Find all functions with per_iter_count > 0
            for function_name, args_data in function_data.items():
                if function_name == 'Solver_iteration':
                    continue
                    
                for args_key, data in args_data.items():
                    per_iter_count = data.get('per_iter_count', 0)
                    
                    if per_iter_count and per_iter_count > 0:
                        # Find matching benchmark for this function
                        if benchmark_id in data['benchmarks']:
                            function_benchmark_data = data['benchmarks'][benchmark_id]
                            function_time = function_benchmark_data['execution_time_avg']
                            # Skip functions with zero execution time
                            if function_time <= 0:
                                continue
                            # Calculate percentage of solver time
                            percentage = (function_time * per_iter_count / solver_time) * 100.0
                            
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
                            if display_name not in benchmark_functions:
                                benchmark_functions.append(display_name)
                            
                            # Store percentage data
                            benchmark_percentages[display_name] = {
                                'percentage': percentage,
                                'time': function_time,
                                'per_iter': per_iter_count
                            }
            
            # Store data for this benchmark
            all_benchmark_data[benchmark_id] = {
                'footprint': size_footprint,
                'memory_footprint_bytes': memory_footprint_bytes,
                'functions': benchmark_functions,
                'percentages': benchmark_percentages,
                'solver_time': solver_time
            }
        
        # If we have no valid data, skip
        if not all_benchmark_data:
            print(f"No valid data for thread count {thread_count}, skipping")
            continue
        
        # Sort benchmarks by memory footprint for consistent ordering
        sorted_benchmarks = sorted(all_benchmark_data.items(), 
                                 key=lambda x: x[1]['memory_footprint_bytes'])
        
        # Get unique set of all functions across all benchmarks with non-zero percentages
        all_functions = set()
        for benchmark_data in all_benchmark_data.values():
            for func_name, func_data in benchmark_data['percentages'].items():
                if func_data['percentage'] > 0:  # Only include functions with non-zero percentage
                    all_functions.add(func_name)
        
        # Sort functions by their maximum percentage across all benchmarks
        function_max_percentages = {}
        for func in all_functions:
            if func in sorted_benchmarks[-1][1]['percentages']:
                function_max_percentages[func] = sorted_benchmarks[-1][1]['percentages'][func]['percentage']
            else: 
                function_max_percentages[func] = 0
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
        x_labels = []
        x_positions = []
        bottom_values = np.zeros(len(sorted_benchmarks))
        
        # Create x-axis labels with memory size
        for i, (benchmark_id, benchmark_data) in enumerate(sorted_benchmarks):
            footprint = benchmark_data['footprint']
            formatted_footprint = format_size(footprint)
            x_labels.append(formatted_footprint)  # Only the value, no "Memory:" prefix
            x_positions.append(i)
        
        # Plot stacked bars for each function
        handles = []  # For legend
        labels = []   # For legend
        
        # Add "Total (μs)" annotation near the top-right of the y-axis
        plt.annotate("Total (μs)", xy=(0, 1.05), xycoords=('axes fraction', 'axes fraction'),
                     ha='left', va='center', fontsize=9)
        
        for func in sorted_functions:
            values = []
            for benchmark_id, benchmark_data in sorted_benchmarks:
                if func in benchmark_data['percentages']:
                    values.append(benchmark_data['percentages'][func]['percentage'])
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
                    benchmark_id, benchmark_data = sorted_benchmarks[i]
                    func_data = benchmark_data['percentages'].get(func, {})
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
        for i, (benchmark_id, benchmark_data) in enumerate(sorted_benchmarks):
            solver_time_us = benchmark_data['solver_time'] * 1e6
            plt.text(x_positions[i], 105, f"{solver_time_us:.1f}", 
                     ha='center', va='bottom', fontsize=9)
        
        # Set up the plot
        plt.ylabel('Percentage of Solver Iteration Time (%)')
        plt.xlabel('Memory Footprint')  # Global x-axis label
        plt.title(f'Function Time Distribution per Iteration ({thread_count} threads)', fontsize=14, pad=15)  # Added padding
        plt.xticks(x_positions, x_labels, rotation=15, ha='right')  # Rotate labels 15 degrees
        plt.ylim(0, 108)  # Increased slightly for more space at top
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Position the legend below the plot with less space
        legend_cols = len(sorted_functions)  # Max 5 columns, adjust as needed
        
        # Add the legend below the plot with reduced spacing
        plt.legend(handles=handles, labels=labels, 
                  loc='upper center', bbox_to_anchor=(0.5, -0.08),  # Reduced space
                  ncol=legend_cols, fontsize=10)
        
        # Tight layout with room for legend and rotated labels
        #plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Reduced bottom margin for legend
        
        # Save the plot with higher resolution but smaller margins
        plt.savefig(os.path.join(output_dir, f'iteration_percentages_{thread_count}t.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def find_all_thread_directories(results_dir):
    """Find all thread directories (t*) in the results directory."""
    pattern = os.path.join(results_dir, 't*')
    thread_dirs = glob.glob(pattern)
    
    # Extract thread counts and sort
    thread_counts = []
    for thread_dir in thread_dirs:
        basename = os.path.basename(thread_dir)
        match = re.match(r't(\d+)', basename)
        if match:
            thread_counts.append(int(match.group(1)))
    
    return sorted(thread_counts)


def collect_all_results_multi_thread(results_dir, filegroup_name, clearun_dir=None):
    """
    Collect results from all analysis files across ALL thread counts for a specific filegroup.
    
    Returns:
        Dictionary organized as:
        {
            thread_count: {
                function_name: {
                    simplified_args: {
                        'operator_info': str,
                        'thread_count': int,
                        'per_iter_count': int,
                        'benchmarks': {
                            benchmark_id: {
                                'benchmark_id': str,
                                'memory_footprint_str': str,
                                'memory_footprint_bytes': float,
                                'execution_time_avg': float,
                                'execution_time_min': float,
                                'execution_time_max': float,
                                'execution_time_stddev': float,
                                'actual_execution_time': float,  # Added for actual execution time
                                'models': [...]
                            }
                        }
                    }
                }
            }
        }
    """
    # Find all thread directories
    thread_counts = find_all_thread_directories(results_dir)
    
    if not thread_counts:
        print(f"Error: No thread directories (t*) found in: {results_dir}")
        exit(1)
    
    print(f"Found {len(thread_counts)} thread configurations: {thread_counts}")
    
    # Collect data for each thread count
    all_thread_data = {}
    
    for threads in thread_counts:
        print(f"\n{'='*80}")
        print(f"Processing thread count: {threads}")
        print(f"{'='*80}")
        
        # Construct the path: results_dir/t{threads}/results/analysis/{filegroup_name}
        analysis_dir = os.path.join(
            results_dir, f't{threads}', 'results', 'analysis', f'{filegroup_name}')

        if not os.path.exists(analysis_dir):
            print(f"Warning: Analysis directory does not exist: {analysis_dir}")
            print(f"Skipping thread count {threads}")
            continue

        # Find all files matching the filegroup pattern
        pattern = os.path.join(analysis_dir, f'*_analysis.log')
        all_files = glob.glob(pattern)
        
        if not all_files:
            print(f"Warning: No analysis files found matching pattern: {pattern}")
            print(f"Skipping thread count {threads}")
            continue
        
        print(f"Found {len(all_files)} analysis files for filegroup '{filegroup_name}' with {threads} threads")
        
        # Group data by function -> simplified_args -> benchmark_file
        grouped_data = {}
        
        for i, filepath in enumerate(all_files):
            print(f"  Parsing ({i+1}/{len(all_files)}): {os.path.basename(filepath)}")
            
            # Extract benchmark identifier from filename (remove _analysis.log suffix)
            benchmark_id = os.path.basename(filepath).replace('_analysis.log', '')
            
            file_results = parse_analysis_file(filepath)
            
            # Skip if file_results is empty
            if not file_results:
                print(f"    Warning: No valid data found")
                continue
            
            # Try to get actual execution time from output.log file if clearun_dir is provided
            actual_execution_time = None
            if clearun_dir:
                output_filepath = os.path.join(clearun_dir, f't{threads}', 'outputs', filegroup_name, f'{benchmark_id}_output.log')
                if os.path.exists(output_filepath):
                    actual_execution_time = parse_output_log_file(output_filepath)
                    if actual_execution_time is not None:
                        print(f"    Found actual execution time: {actual_execution_time:.6f}s for {benchmark_id}")
                else:
                    print(f"    Warning: Output file not found: {output_filepath}")
            
            for function_name, args_data in file_results.items():
                # Initialize function in data structure if needed
                if function_name not in grouped_data:
                    grouped_data[function_name] = {}
                
                for args, metrics in args_data.items():
                    # Create a key that removes size information
                    simplified_args = simplify_args(args)
                    
                    # Initialize simplified args in data structure if needed
                    if simplified_args not in grouped_data[function_name]:
                        grouped_data[function_name][simplified_args] = {
                            'operator_info': metrics.get('operator_info', ''),
                            'thread_count': metrics['thread_count'],
                            'per_iter_count': metrics.get('per_iter_count', 0),
                            'benchmarks': {}  # Group by benchmark file
                        }
                    
                    # Get memory footprint from the first model (should be same for all models)
                    memory_footprint_str = metrics.get('expected_footprint', '0 B')
                    memory_footprint_bytes = parse_memory_footprint(memory_footprint_str)
                    
                    # Store data for this benchmark
                    grouped_data[function_name][simplified_args]['benchmarks'][benchmark_id] = {
                        'benchmark_id': benchmark_id,
                        'memory_footprint_str': memory_footprint_str,
                        'memory_footprint_bytes': memory_footprint_bytes,
                        'execution_time_avg': metrics['execution_time_avg'],
                        'execution_time_min': metrics['execution_time_min'],
                        'execution_time_max': metrics['execution_time_max'],
                        'execution_time_stddev': metrics['execution_time_stddev'],
                        'actual_execution_time': actual_execution_time,  # Add actual execution time if available
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
                        grouped_data[function_name][simplified_args]['benchmarks'][benchmark_id]['models'].append(model_data)
        
        # Store grouped data for this thread count
        if grouped_data:
            all_thread_data[threads] = grouped_data
            
            # Print summary for this thread count
            print(f"\n  Summary for {threads} threads:")
            print(f"  Total functions found: {len(grouped_data)}")
            for func_name in sorted(grouped_data.keys())[:5]:  # Show first 5 functions
                args_count = len(grouped_data[func_name])
                total_benchmarks = sum(len(data['benchmarks']) for data in grouped_data[func_name].values())
                print(f"    {func_name}: {args_count} arg variations, {total_benchmarks} total benchmarks")
            if len(grouped_data) > 5:
                print(f"    ... and {len(grouped_data) - 5} more functions")
    
    return all_thread_data


def print_summary(all_thread_data):
    """Print a comprehensive summary of all collected data."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SUMMARY - ALL THREAD COUNTS")
    print(f"{'='*80}")
    
    if not all_thread_data:
        print("No data collected!")
        return
    
    # Get all unique functions across all thread counts
    all_functions = set()
    for thread_data in all_thread_data.values():
        all_functions.update(thread_data.keys())
    
    print(f"\nTotal thread configurations: {len(all_thread_data)}")
    print(f"Thread counts: {sorted(all_thread_data.keys())}")
    print(f"Total unique functions across all threads: {len(all_functions)}")
    
    # For each function, show data across thread counts
    print(f"\n{'='*80}")
    print("FUNCTION ANALYSIS ACROSS THREAD COUNTS")
    print(f"{'='*80}")
    
    for function_name in sorted(all_functions):
        print(f"\nFunction: {function_name}")
        print(f"-" * 80)
        
        # Show which thread counts have this function
        available_threads = [t for t in sorted(all_thread_data.keys()) if function_name in all_thread_data[t]]
        print(f"  Available in thread counts: {available_threads}")
        
        # For each thread count, show argument variations and benchmark counts
        for threads in available_threads:
            function_data = all_thread_data[threads][function_name]
            print(f"\n  Thread count: {threads}")
            
            for simplified_args, data in function_data.items():
                num_benchmarks = len(data['benchmarks'])
                operator_info = data['operator_info']
                per_iter = data['per_iter_count']
                
                # Get memory footprint range
                if data['benchmarks']:
                    footprints = [b['memory_footprint_bytes'] for b in data['benchmarks'].values()]
                    min_fp = min(footprints) / 1024  # KB
                    max_fp = max(footprints) / 1024  # KB
                    
                    fp_str = f"{min_fp:.1f} KB - {max_fp:.1f} KB" if min_fp != max_fp else f"{min_fp:.1f} KB"
                else:
                    fp_str = "N/A"
                
                op_str = f" ({operator_info})" if operator_info else ""
                iter_str = f" [per_iter: {per_iter}]" if per_iter else ""
                
                print(f"    {simplified_args}{op_str}{iter_str}")
                print(f"      Benchmarks: {num_benchmarks}, Memory footprint range: {fp_str}")
                
                # Show example execution times
                if data['benchmarks']:
                    example_benchmark = list(data['benchmarks'].values())[0]
                    print(f"      Example execution time: {example_benchmark['execution_time_avg']:.4e} s")
                    print(f"      Example models: {len(example_benchmark['models'])} models")


def plot_best_performance_across_threads(all_thread_data, output_dir="plots"):
    """
    Create plots showing the best (minimum) execution time across all thread counts.
    For each memory footprint, shows which thread count achieved the best performance.
    
    Args:
        all_thread_data: Dictionary with structure {thread_count: {function_name: {simplified_args: {...}}}}
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all unique functions across all thread counts
    all_functions = set()
    for thread_data in all_thread_data.values():
        all_functions.update(thread_data.keys())
    
    # Define markers to use
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    
    # Create a color palette
    colors = plt.cm.tab10.colors
    
    # Process each function
    for function_name in sorted(all_functions):
        # Collect all argument variations across all threads for this function
        all_args_variations = set()
        for thread_count, thread_data in all_thread_data.items():
            if function_name in thread_data:
                all_args_variations.update(thread_data[function_name].keys())
        
        if not all_args_variations:
            continue
        
        # Group function variations by argument type signature and operator
        arg_categories = {}
        
        for args_key in all_args_variations:
            # Get operator info from first available thread
            operator_info = ""
            for thread_count, thread_data in all_thread_data.items():
                if function_name in thread_data and args_key in thread_data[function_name]:
                    operator_info = thread_data[function_name][args_key].get('operator_info', '')
                    break
            
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
            
            # More specific matching for function types
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
            fig = plt.figure(figsize=(10, 8))
            grid_cols = 1
            grid_rows = 1
        else:
            grid_cols = min(3, num_categories)
            grid_rows = (num_categories + grid_cols - 1) // grid_cols
            fig = plt.figure(figsize=(6*grid_cols, 5*grid_rows))
        
        # Process each category in a separate subplot
        for idx, (category_key, args_keys) in enumerate(arg_categories.items()):
            ax = fig.add_subplot(grid_rows, grid_cols, idx+1)
            
            # Add subplot title
            subplot_title = f"{function_name}"
            if category_key != "default":
                display_category = category_key.replace("_", " ").replace("vector", "Vector")
                display_category = display_category.replace("scalar", "Scalar").replace("monoid", "Monoid")
                subplot_title += f" ({display_category})"
            ax.set_title(subplot_title)
            
            # Keep track of lines for the legend
            all_lines = []
            all_labels = []
            
            # Process each argument variation in this category
            for i, args_key in enumerate(args_keys):
                # Aggregate data across all thread counts for this args_key
                # Structure: {memory_footprint_bytes: {thread_count: execution_time}}
                footprint_data = defaultdict(dict)
                # Structure: {memory_footprint_bytes: {thread_count: cost_prediction}}
                footprint_cost_data = defaultdict(dict)
                
                # Collect data from all thread counts
                for thread_count, thread_data in all_thread_data.items():
                    if function_name not in thread_data or args_key not in thread_data[function_name]:
                        continue
                    
                    data = thread_data[function_name][args_key]
                    
                    for benchmark_id, benchmark_data in data['benchmarks'].items():
                        memory_footprint_bytes = benchmark_data['memory_footprint_bytes']
                        execution_time_avg = benchmark_data['execution_time_avg']
                        
                        # Skip functions with zero execution time
                        if execution_time_avg <= 0 or memory_footprint_bytes <= 0:
                            continue
                        
                        footprint_data[memory_footprint_bytes][thread_count] = execution_time_avg
                        
                        # Also collect cost predictions for all threads
                        for model in benchmark_data['models']:
                            model_name = model.get('model_name', '')
                            if 'k-multi-bsp' in model_name.lower():
                                cost = model.get('cost', None)
                                if cost is not None and cost > 0:
                                    footprint_cost_data[memory_footprint_bytes][thread_count] = cost
                                break
                
                # Skip if no valid data points
                if not footprint_data:
                    continue
                
                # For each footprint, find the best (minimum) execution time and which thread achieved it
                footprints = []
                best_times = []
                best_threads = []
                predicted_costs = []
                cost_model_name = None  # Track which model we're using
                
                for memory_footprint_bytes in sorted(footprint_data.keys()):
                    thread_times = footprint_data[memory_footprint_bytes]
                    
                    # Find the thread with minimum execution time
                    best_thread = min(thread_times.keys(), key=lambda t: thread_times[t])
                    best_time = thread_times[best_thread]
                    
                    memory_kb = memory_footprint_bytes / 1024.0
                    footprints.append(memory_kb)
                    best_times.append(best_time)
                    best_threads.append(best_thread)
                    
                    # Get predicted cost from k-multi-BSP model for this thread and footprint
                    predicted_cost = None
                    if function_name in all_thread_data[best_thread] and args_key in all_thread_data[best_thread][function_name]:
                        data = all_thread_data[best_thread][function_name][args_key]
                        for benchmark_id, benchmark_data in data['benchmarks'].items():
                            if benchmark_data['memory_footprint_bytes'] == memory_footprint_bytes:
                                # Find k-multi-BSP model (case-insensitive)
                                for model in benchmark_data['models']:
                                    model_name = model.get('model_name', '')
                                    if 'k-multi-bsp' in model_name.lower():
                                        predicted_cost = model.get('cost', None)
                                        if cost_model_name is None:
                                            cost_model_name = model_name  # Save the actual model name
                                        break
                                break
                    
                    predicted_costs.append(predicted_cost)
                
                # Skip if no valid data points
                if not footprints:
                    continue
                
                # Choose a color and marker for this function variation
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                # Get per_iter_count for display name
                per_iter_count = None
                for thread_count, thread_data in all_thread_data.items():
                    if function_name in thread_data and args_key in thread_data[function_name]:
                        per_iter_count = thread_data[function_name][args_key].get('per_iter_count')
                        break
                
                # Create a simplified display name
                display_name = "Best Performance"
                if per_iter_count is not None and per_iter_count > 0:
                    display_name += f" [{per_iter_count}/iter]"
                
                # Plot the best performance line
                time_line, = ax.plot(footprints, best_times, '-', marker=marker, color=color, 
                                    label=display_name, linewidth=2, markersize=8)
                
                # Add time line to legend
                all_lines.append(time_line)
                all_labels.append(display_name)
                
                # Plot predicted cost line (k-multi-BSP model) if available
                valid_predicted_costs = [c for c in predicted_costs if c is not None and c > 0]
                if valid_predicted_costs and cost_model_name:
                    # Filter to only plot points where we have predictions
                    pred_footprints = []
                    pred_costs = []
                    for fp, cost in zip(footprints, predicted_costs):
                        if cost is not None and cost > 0:
                            pred_footprints.append(fp)
                            pred_costs.append(cost)
                    
                    if pred_footprints:
                        cost_label = f"{cost_model_name} cost for Best"
                        cost_line, = ax.plot(pred_footprints, pred_costs, '--', marker=marker, color=color,
                                            label=cost_label, linewidth=2, markersize=6, alpha=0.7)
                        all_lines.append(cost_line)
                        all_labels.append(cost_label)
                
                # For Solver_iteration, add actual execution time as a green line if available
                # Only show actual execution times for the best-performing threads (same as blue line)
                if function_name == 'Solver_iteration':
                    actual_times = []
                    actual_footprints = []
                    actual_threads = []
                    
                    # Use the same footprints and best_threads that were determined for the blue line
                    for j, (fp, best_thread) in enumerate(zip(footprints, best_threads)):
                        # Get actual execution time for this best-performing thread and footprint
                        if best_thread in all_thread_data and function_name in all_thread_data[best_thread]:
                            function_data = all_thread_data[best_thread][function_name]
                            if args_key in function_data:
                                args_data = function_data[args_key]
                                if 'benchmarks' in args_data:
                                    for benchmark_id, benchmark_data in args_data['benchmarks'].items():
                                        if (benchmark_data['memory_footprint_bytes'] / 1024.0 == fp and 
                                            benchmark_data.get('actual_execution_time') is not None):
                                            actual_times.append(benchmark_data['actual_execution_time'])
                                            actual_footprints.append(fp)
                                            actual_threads.append(best_thread)
                                            break
                    
                    if actual_times:
                        actual_line, = ax.plot(actual_footprints, actual_times, '-', marker='o', color='green', 
                                             linewidth=2, markersize=6, label="Avg Isolated Execution Time [1/iter]")
                        all_lines.append(actual_line)
                        all_labels.append("Avg Isolated Execution Time [1/iter]")
                        
                        # Add thread annotations for the actual execution time points (similar to blue line)
                        for j, (fp, time, thread) in enumerate(zip(actual_footprints, actual_times, actual_threads)):
                            # Only annotate every few points if there are many to avoid clutter
                            if len(actual_footprints) <= 10 or j % max(1, len(actual_footprints) // 10) == 0 or j == len(actual_footprints) - 1:
                                ax.annotate(f't{thread}', 
                                           (fp, time),
                                           xytext=(0, -8),  # Negative offset to place below the point
                                           textcoords='offset points',
                                           ha='center',
                                           va='top',
                                           fontsize=10,
                                           color='green',
                                           weight='bold')
                
                # Plot the lowest cost prediction across ALL threads (red dotted line)
                if footprint_cost_data:
                    best_cost_footprints = []
                    best_costs = []
                    best_cost_threads = []
                    best_cost_actual_times = []  # Actual execution times for the predicted-best threads
                    
                    for memory_footprint_bytes in sorted(footprint_cost_data.keys()):
                        thread_costs = footprint_cost_data[memory_footprint_bytes]
                        
                        if not thread_costs:
                            continue
                        
                        # Find the thread with minimum cost prediction
                        best_cost_thread = min(thread_costs.keys(), key=lambda t: thread_costs[t])
                        best_cost = thread_costs[best_cost_thread]
                        
                        # Get the actual execution time for this predicted-best thread
                        actual_time = footprint_data[memory_footprint_bytes].get(best_cost_thread, None)
                        
                        memory_kb = memory_footprint_bytes / 1024.0
                        best_cost_footprints.append(memory_kb)
                        best_costs.append(best_cost)
                        best_cost_threads.append(best_cost_thread)
                        best_cost_actual_times.append(actual_time)
                    
                    if best_cost_footprints and cost_model_name:
                        # Plot red dotted line (predicted costs)
                        best_cost_label = f"{cost_model_name} Lowest cost"
                        best_cost_line, = ax.plot(best_cost_footprints, best_costs, ':', marker=marker, color='red',
                                                  label=best_cost_label, linewidth=2, markersize=6, alpha=0.8)
                        all_lines.append(best_cost_line)
                        all_labels.append(best_cost_label)
                        
                        # Plot red solid line (actual execution times for predicted-best threads)
                        actual_time_footprints = []
                        actual_times = []
                        for fp, time in zip(best_cost_footprints, best_cost_actual_times):
                            if time is not None and time > 0:
                                actual_time_footprints.append(fp)
                                actual_times.append(time)
                        
                        if actual_time_footprints:
                            actual_time_label = f"Actual time for Lowest cost"
                            actual_time_line, = ax.plot(actual_time_footprints, actual_times, '-', marker=marker, color='red',
                                                        label=actual_time_label, linewidth=2, markersize=8, alpha=0.8)
                            all_lines.append(actual_time_line)
                            all_labels.append(actual_time_label)
                        
                        # Annotate each point with the thread count that achieved lowest cost
                        for j, (fp, cost, thread) in enumerate(zip(best_cost_footprints, best_costs, best_cost_threads)):
                            # Only annotate every few points if there are many to avoid clutter
                            if len(best_cost_footprints) <= 10 or j % max(1, len(best_cost_footprints) // 10) == 0 or j == len(best_cost_footprints) - 1:
                                ax.annotate(f't{thread}', 
                                           (fp, cost),
                                           xytext=(0, -8),  # Negative offset to place below the point
                                           textcoords='offset points',
                                           ha='center',
                                           va='top',
                                           fontsize=10,
                                           color='red',
                                           weight='bold')
                
                # Annotate each point with the thread count that achieved it
                for j, (fp, time, thread) in enumerate(zip(footprints, best_times, best_threads)):
                    # Only annotate every few points if there are many to avoid clutter
                    if len(footprints) <= 10 or j % max(1, len(footprints) // 10) == 0 or j == len(footprints) - 1:
                        ax.annotate(f't{thread}', 
                                   (fp, time),
                                   xytext=(0, 5),
                                   textcoords='offset points',
                                   ha='center',
                                   va='bottom',
                                   fontsize=10,
                                   color=color,
                                   weight='bold')
            
            # Add vertical lines for cache sizes
            for cache_name, cache_size in CACHE_SIZES.items():
                cache_kb = cache_size / 1024.0
                ax.axvline(x=cache_kb, color='gray', linestyle='--', alpha=0.7)
                
                if cache_size < 1024 * 1024:
                    label = f"{cache_name} ({cache_kb:.0f} KB)"
                else:
                    label = f"{cache_name} ({cache_kb/1024:.0f} MB)"
                
                ax.annotate(label, 
                           (cache_kb, ax.get_ylim()[0] * 1.1),
                           xytext=(0, 10),
                           textcoords="offset points",
                           ha='center',
                           va='bottom',
                           fontsize=8,
                           rotation=90,
                           color='black')
            
            # Configure axes
            ax.set_xlabel('Memory Footprint (KB)')
            ax.set_ylabel('Time / Cost (seconds)', color='black')
            
            # Set to log scale
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            
            # Grid
            ax.grid(True, which="both", ls="--", alpha=0.3)
            
            # Add legend for this subplot
            if all_lines:
                if num_categories == 1:
                    ax.legend(all_lines, all_labels, loc='upper left', fontsize=9)
                else:
                    ax.legend(all_lines, all_labels, loc='upper left', fontsize=7)
        
        # Add a main title for the whole figure
        plt.suptitle(f'{function_name} - Best Performance Across All Threads', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure with high resolution
        plt.savefig(os.path.join(output_dir, f'{function_name}_best_across_threads.png'), dpi=150)
        plt.close()
        
        print(f"  Generated plot: {function_name}_best_across_threads.png")


def plot_cost_percentages(thread_data, output_dir="plots"):
    """
    Create a stacked bar chart showing what percentage of the total predicted cost
    is taken by each function that runs per iteration, for all benchmarks.
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
        luminance = (0.299 * bg_color[0] + 0.587 *
                     bg_color[1] + 0.114 * bg_color[2])

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

    # For each thread count
    for thread_count, function_data in thread_data.items():
        # Check if we have Solver_iteration data
        if 'Solver_iteration' not in function_data:
            print(f"Warning: No Solver_iteration data found for {thread_count} threads, skipping cost percentage plot")
            continue
        
        # Get Solver_iteration data
        solver_data = function_data['Solver_iteration']
        solver_args = next(iter(solver_data))
        solver_data_point = solver_data[solver_args]
        
        # Create a dictionary to store data for all benchmarks
        all_benchmark_data = {}
        
        # Process each benchmark for Solver_iteration
        for benchmark_id, solver_benchmark_data in sorted(solver_data_point['benchmarks'].items()):
            # Get solver cost from first available model
            solver_cost = None
            for model in solver_benchmark_data['models']:
                if model.get('cost') is not None:
                    solver_cost = model.get('cost', 0.0)
                    break
            
            if solver_cost is None or solver_cost <= 0:
                print(f"Warning: Invalid Solver_iteration cost for benchmark {benchmark_id}, skipping")
                continue
            
            # Initialize storage for this benchmark
            benchmark_functions = []
            benchmark_percentages = {}
            memory_footprint_bytes = solver_benchmark_data['memory_footprint_bytes']
            size_footprint = memory_footprint_bytes / 1024.0  # KB
            
            # Find all functions with per_iter_count > 0 (excluding Solver_iteration)
            for function_name, args_data in function_data.items():
                if function_name == 'Solver_iteration':
                    continue
                    
                for args_key, data in args_data.items():
                    per_iter_count = data.get('per_iter_count', 0)
                    
                    if per_iter_count and per_iter_count > 0:
                        # Find matching benchmark for this function
                        if benchmark_id in data['benchmarks']:
                            function_benchmark_data = data['benchmarks'][benchmark_id]
                            function_time = function_benchmark_data['execution_time_avg']
                            
                            # Skip functions with zero execution time
                            if function_time <= 0:
                                continue
                            
                            # Get cost from first available model
                            model_cost = None
                            model_name = None
                            for model in function_benchmark_data['models']:
                                if model.get('cost') is not None:
                                    model_cost = model.get('cost', 0.0)
                                    model_name = model.get('model_name', '')
                                    break
                            
                            if model_cost is not None:
                                # Calculate percentage of solver cost
                                function_cost = model_cost * per_iter_count
                                percentage = (function_cost / solver_cost) * 100.0
                                
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
                                if display_name not in benchmark_functions:
                                    benchmark_functions.append(display_name)
                                
                                # Store percentage data
                                benchmark_percentages[display_name] = {
                                    'percentage': percentage,
                                    'cost': model_cost,
                                    'total_cost': function_cost,
                                    'per_iter': per_iter_count,
                                    'model': model_name
                                }
            
            # Store data for this benchmark
            all_benchmark_data[benchmark_id] = {
                'footprint': size_footprint,
                'memory_footprint_bytes': memory_footprint_bytes,
                'functions': benchmark_functions,
                'percentages': benchmark_percentages,
                'solver_cost': solver_cost
            }

        # If we have no valid data, skip
        if not all_benchmark_data:
            print(
                f"No valid cost data for thread count {thread_count}, skipping")
            continue

        # Sort benchmarks by memory footprint for consistent ordering
        sorted_benchmarks = sorted(all_benchmark_data.items(),
                                   key=lambda x: x[1]['memory_footprint_bytes'])

        # Get unique set of all functions across all benchmarks with non-zero percentages
        all_functions = set()
        for benchmark_data in all_benchmark_data.values():
            for func_name, func_data in benchmark_data['percentages'].items():
                # Only include functions with non-zero percentage
                if func_data['percentage'] > 0:
                    all_functions.add(func_name)

        # Sort functions by their maximum percentage across all benchmarks
        function_max_percentages = {}
        for func in all_functions:
            if func in sorted_benchmarks[-1][1]['percentages']:
                function_max_percentages[func] = sorted_benchmarks[-1][1]['percentages'][func]['percentage']
            else: 
                function_max_percentages[func] = 0
        # Sort functions by max percentage (descending)
        sorted_functions = sorted(
            all_functions, key=lambda f: function_max_percentages[f], reverse=True)

        # Create consistent colors for each function
        color_map = {}
        colormap = plt.cm.viridis
        for i, func in enumerate(sorted_functions):
            color_map[func] = colormap(i / max(1, len(sorted_functions) - 1))

        # Create figure with more width for the actual plot
        plt.figure(figsize=(16, 8))

        # Prepare data for plotting
        x_labels = []
        x_positions = []
        bottom_values = np.zeros(len(sorted_benchmarks))

        # Create x-axis labels with memory size
        for i, (benchmark_id, benchmark_data) in enumerate(sorted_benchmarks):
            footprint = benchmark_data['footprint']
            formatted_footprint = format_size(footprint)
            # Only the value, no "Memory:" prefix
            x_labels.append(formatted_footprint)
            x_positions.append(i)

        # Plot stacked bars for each function
        handles = []  # For legend
        labels = []   # For legend

        # Add "Total Cost" annotation near the top-left of the y-axis
        plt.annotate("Total Cost", xy=(0, 1.05), xycoords=('axes fraction', 'axes fraction'),
                     ha='left', va='center', fontsize=9)

        for func in sorted_functions:
            values = []
            for benchmark_id, benchmark_data in sorted_benchmarks:
                if func in benchmark_data['percentages']:
                    values.append(
                        benchmark_data['percentages'][func]['percentage'])
                else:
                    values.append(0)

            # Skip function if all values are zero
            if all(v == 0 for v in values):
                continue

            # Plot this function's bars
            bar = plt.bar(x_positions, values, bottom=bottom_values,
                          label=func, color=color_map[func])

            # Store for legend
            handles.append(bar)
            labels.append(func)

            # Add percentage labels inside the bars if they're large enough
            for i, v in enumerate(values):
                if v >= 5.0:  # Only show label if percentage is at least 5%
                    # Get data for annotation
                    benchmark_id, benchmark_data = sorted_benchmarks[i]
                    func_data = benchmark_data['percentages'].get(func, {})
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

        # Add solver cost above each bar
        for i, (benchmark_id, benchmark_data) in enumerate(sorted_benchmarks):
            solver_cost = benchmark_data['solver_cost']
            plt.text(x_positions[i], 105, f"{solver_cost:.2e}",
                     ha='center', va='bottom', fontsize=9)

        # Set up the plot
        plt.ylabel('Percentage of Total Predicted Cost (%)')
        plt.xlabel('Memory Footprint')  # Global x-axis label
        plt.title(
            f'Function Cost Distribution per Iteration ({thread_count} threads)', fontsize=14, pad=15)
        plt.xticks(x_positions, x_labels, rotation=15,
                   ha='right')  # Rotate labels 15 degrees
        plt.ylim(0, 108)  # Leave room for total cost label
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Position the legend below the plot
        legend_cols = len(sorted_functions)
        plt.legend(handles=handles, labels=labels,
                   loc='upper center', bbox_to_anchor=(0.5, -0.08),
                   ncol=legend_cols, fontsize=10)

        # Tight layout
        # plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save the plot
        plt.savefig(os.path.join(output_dir, f'cost_percentages_{thread_count}t.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Plot performance data from analysis files.')
    parser.add_argument('--results-dir', default='results', 
                        help='Directory containing the analysis files (default: results)')
    parser.add_argument('--threads', default='1',
                        help='Thread count to plot (default: 1). Use "all" to analyze all available thread counts.')
    parser.add_argument('--filegroup-name', required=True,
                        help='Name of the filegroup (e.g., banded_diag)')
    parser.add_argument('--clearun-dir', default=None,
                        help='Directory containing output.log files for actual execution time measurements (optional)')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    filegroup_name = args.filegroup_name
    threads_arg = args.threads
    clearun_dir = args.clearun_dir
    
    # Check if we should analyze all thread counts
    if threads_arg.lower() == 'all':
        print(f"Multi-Thread Analysis Mode")
        print(f"Results directory: {results_dir}")
        print(f"Filegroup: {filegroup_name}")
        print(f"{'='*80}\n")
        
        # Collect results across all thread counts
        all_thread_data = collect_all_results_multi_thread(results_dir, filegroup_name, clearun_dir)
        
        if not all_thread_data:
            print(f"\nNo data found for filegroup '{filegroup_name}'")
            exit(1)
        
        # Print comprehensive summary
        print_summary(all_thread_data)
        
        # Create output directory for multi-thread plots
        plots_base_dir = os.path.join(results_dir, 'results', 'plots', f'{filegroup_name}_all_threads')
        os.makedirs(plots_base_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Generating aggregated plots in: {plots_base_dir}")
        print(f"{'='*80}\n")
        
        # Generate plots showing best performance across all threads
        plot_best_performance_across_threads(all_thread_data, plots_base_dir)
        
        print(f"\n{'='*80}")
        print("Analysis and plotting complete!")
        print(f"{'='*80}")
    else:
        # Single thread mode (original behavior)
        try:
            threads = int(threads_arg)
        except ValueError:
            print(f"Error: --threads must be an integer or 'all', got: {threads_arg}")
            exit(1)
        
        # Collect results for the specified filegroup and thread count
        function_data = collect_all_results(results_dir, filegroup_name, threads, clearun_dir)
        
        if not function_data:
            print(f"No data found for filegroup '{filegroup_name}' with {threads} threads")
            exit(1)
        
        function_count = len(function_data)
        print(f"Found data for {function_count} distinct function variations")
        
        # Create output directory
        plots_base_dir = os.path.join(
            results_dir, f't{threads}', 'results', 'plots', f'{filegroup_name}')
        os.makedirs(plots_base_dir, exist_ok=True)
        print(f"Generating plots in: {plots_base_dir}")
        
        # Generate plots
        plot_results(function_data, plots_base_dir)
        print(f"Generating time percentage plots in: {plots_base_dir}")
        plot_iteration_percentages({threads: function_data}, plots_base_dir)
        print(f"Generating cost percentage plots in: {plots_base_dir}")
        plot_cost_percentages({threads: function_data}, plots_base_dir)
        print(f"Plotting complete for {threads} threads!")
        
        print("All plotting tasks completed!")

if __name__ == "__main__":
    main()
