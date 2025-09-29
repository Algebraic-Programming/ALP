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


def collect_all_results(results_dir, filegroup_name, threads):
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
                    
                    if memory_footprint_bytes > 0:
                        # Convert to KB for better scale
                        memory_kb = memory_footprint_bytes / 1024.0
                        
                        # Add execution time data point
                        footprints.append(memory_kb)
                        exec_times_avg.append(benchmark_data['execution_time_avg'])
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
    is taken by each function that runs per iteration, for all matrix sizes.
    """
    # TODO: Implement iteration percentage plotting
    pass

def plot_cost_percentages(thread_data, output_dir="plots"):
    """
    Create a stacked bar chart showing what percentage of the total predicted cost
    is taken by each function that runs per iteration, for all matrix sizes.
    """
    # TODO: Implement cost percentage plotting
    pass

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Plot performance data from analysis files.')
    parser.add_argument('--results-dir', default='results', 
                        help='Directory containing the analysis files (default: results)')
    parser.add_argument('--threads', type=int, default=1,
                        help='Only plot results for this thread count (default: all)')
    parser.add_argument('--filegroup-name', required=True,
                        help='Name of the filegroup (e.g., banded_diag)')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    filegroup_name = args.filegroup_name
    threads = args.threads
    
    # Collect results for the specified filegroup and thread count
    function_data = collect_all_results(results_dir, filegroup_name, threads)
    
    if not function_data:
        print(f"No data found for filegroup '{filegroup_name}' with {threads} threads")
        exit(1)
    
    function_count = len(function_data)
    print(f"Found data for {function_count} distinct function variations")
    
    # Create output directory
    plots_base_dir = os.path.join(results_dir, f't{threads}', 'results', 'plots')
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
