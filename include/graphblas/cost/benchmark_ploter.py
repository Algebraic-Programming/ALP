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
    analysis_dir = os.path.join(results_dir, f't{threads}', 'results', 'analysis')
    
    if not os.path.exists(analysis_dir):
        print(f"Error: Analysis directory does not exist: {analysis_dir}")
        exit(1)
    
    # Find all files matching the filegroup pattern
    pattern = os.path.join(analysis_dir, f'{filegroup_name}*_analysis.log')
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"Error: No analysis files found matching pattern: {pattern}")
        exit(1)
    
    print(f"Found {len(all_files)} analysis files for filegroup '{filegroup_name}' with {threads} threads")
    
    # Group data by function -> simplified_args -> matrix_size
    grouped_data = {}
    
    print(f"\nStarting to parse {len(all_files)} files...")
    for i, filepath in enumerate(all_files):
        if i == 0:
            print(f"\nParsing: {os.path.basename(filepath)}")
        else:
            print(f"Parsing: {os.path.basename(filepath)}")
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
                        'memory_footprints': {}  # Group by memory footprint
                    }
                
                # Get memory footprint from the first model (should be same for all models)
                memory_footprint_str = metrics.get('expected_footprint', '0 B')
                memory_footprint_bytes = parse_memory_footprint(memory_footprint_str)
                
                # Store data for this memory footprint
                grouped_data[function_name][simplified_args]['memory_footprints'][memory_footprint_bytes] = {
                    'memory_footprint_str': memory_footprint_str,
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
                    grouped_data[function_name][simplified_args]['memory_footprints'][memory_footprint_bytes]['models'].append(model_data)
    
    # Print final grouping summary
    print(f"\n=== FINAL GROUPING SUMMARY ===")
    print(f"Total functions found: {len(grouped_data)}")
    for func_name, args_data in grouped_data.items():
        print(f"  {func_name}:")
        for simplified_args, data in args_data.items():
            memory_footprints = sorted(data['memory_footprints'].keys())
            print(f"    {simplified_args}: {len(memory_footprints)} memory footprints {[f'{fp//1024}KB' for fp in memory_footprints]}")
    
    # Print a single example of internal data structure
    print(f"\n=== EXAMPLE INTERNAL DATA STRUCTURE ===")
    if grouped_data:
        first_func = list(grouped_data.keys())[0]
        first_args = list(grouped_data[first_func].keys())[0]
        first_memory_footprint = list(grouped_data[first_func][first_args]['memory_footprints'].keys())[0]
        
        example_data = grouped_data[first_func][first_args]['memory_footprints'][first_memory_footprint]
        
        print(f"Function: {first_func}")
        print(f"Arguments: {first_args}")
        print(f"Memory footprint: {example_data['memory_footprint_str']}")
        print(f"Execution time (avg): {example_data['execution_time_avg']:.2e}s")
        print(f"Models found: {len(example_data['models'])}")
        for i, model in enumerate(example_data['models']):  # Show all models
            print(f"  Model {i+1}: {model['model_name']} - Cost: {model['cost']:.2e}, Footprint: {model['footprint']}")
    
    return grouped_data

def plot_results(function_data, output_dir="plots"):
    """Create plots for each function's performance metrics with different argument types in subplots."""
    # TODO: Implement plotting logic
    pass

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
    parser.add_argument('--threads', type=int, default=None,
                        help='Only plot results for this thread count (default: all)')
    parser.add_argument('--filegroup-name', required=True,
                        help='Name of the filegroup (e.g., banded_diag)')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    filegroup_name = args.filegroup_name
    threads = args.threads
    
    # If --threads is not specified, try to find available thread counts
    if threads is None:
        # Look for t* directories in results_dir
        thread_dirs = glob.glob(os.path.join(results_dir, 't*'))
        if not thread_dirs:
            print(f"Error: No thread directories found in {results_dir}")
            exit(1)
        
        # Extract thread counts
        thread_counts = []
        for thread_dir in thread_dirs:
            match = re.search(r'/t(\d+)$', thread_dir)
            if match:
                thread_counts.append(int(match.group(1)))
        
        if not thread_counts:
            print(f"Error: Could not determine thread counts from directories in {results_dir}")
            exit(1)
        
        thread_counts.sort()
        print(f"Found thread counts: {thread_counts}")
        threads = thread_counts[0]  # Use the first available thread count
        print(f"Using thread count: {threads}")
    
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
