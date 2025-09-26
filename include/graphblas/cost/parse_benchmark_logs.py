#!/usr/bin/env python3
# filepath: /home/panastasiadis/ALP/include/graphblas/cost/cost_and_time_parser_plus_DEBUG.py
"""
Parser for analyzing function call costs and execution times from log files.
"""

import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple, DefaultDict
import numpy as np

# Regular expression patterns
PATTERNS = {
    'function_entry': re.compile(r'\[TRACING\] Entering function: (\w+<[^>]*>|\w+<\d+>|\w+) with (\d+) arguments'),
    'arg_types': re.compile(r'\[TRACING\] Argument types: (.*)'),
    'model_header': re.compile(r'===== (.*) Cost Prediction ====='),  # Updated to match any model header
    'threads': re.compile(r'Threads: (\d+)'),
    'aggregator': re.compile(r'Stream aggregator: (\w+)'),
    'footprint': re.compile(r'Memory footprint: ([\d.]+ [KMGTP]?B)'),
    'level': re.compile(r'Superstep type \d+ \(level (\d+)\):'),
    'model_cost': re.compile(r'Total cost: ([0-9.e+-]+) seconds'),
    # Removed 'cost' pattern
    'exit': re.compile(r'\[TRACING\] Exiting function: (\w+<[^>]*>|\w+<\d+>|\w+) \(took (\d+)μs\)'),
    'cg_iterations': re.compile(r'number of CG iterations: (\d+)')
}

def extract_base_function_name(full_name: str) -> str:
    """Extract the base function name from a potentially templated function name."""
    match = re.match(r'(\w+)', full_name)
    if match:
        return match.group(1)
    return full_name

def extract_cg_iterations(log_file_path: str) -> int:
    """Extract the number of CG iterations from the log file."""
    try:
        with open(log_file_path, 'r') as file:
            content = file.read()
            match = PATTERNS['cg_iterations'].search(content)
            if match:
                return int(match.group(1))
    except Exception as e:
        print(f"Error extracting CG iterations: {e}")
    
    # Default to 10 iterations if not found
    return 10

def parse_log_file(log_file_path: str) -> DefaultDict[str, List[Dict[str, Any]]]:
    """
    Parse the log file and extract function calls with their argument details, models and costs.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        Dictionary with function calls information
    """
    function_data = defaultdict(list)
    
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Match function entry
            match = PATTERNS['function_entry'].search(line)
            if match:
                function_name = match.group(1)
                base_function = extract_base_function_name(function_name)
                
                # Get arguments from the next line
                i += 1
                if i < len(lines) and '[TRACING] Argument types:' in lines[i]:
                    args_line = lines[i].strip()
                    args_match = PATTERNS['arg_types'].search(args_line)
                    if args_match:
                        args = args_match.group(1).strip()
                        
                        # Extract sizes, dimensions from args
                        sizes = re.findall(r'size=(\d+)', args)
                        dimensions = re.findall(r'rows=(\d+),cols=(\d+)', args)
                        nnz = re.findall(r'nnz=(\d+)', args)
                        operators = re.findall(r'operators::(\w+)', args)
                        
                        # Initialize function call data
                        call_data = {
                            'full_name': function_name,
                            'args': args,
                            'sizes': sizes,
                            'dimensions': dimensions,
                            'nnz': nnz,
                            'operators': operators,
                            'models': [],
                            'cost': None,
                            'execution_time': None
                        }
                        
                        # Look for models after the args
                        i += 1
                        while i < len(lines):
                            # Check if we're at the start of a model section
                            if '===== ' in lines[i] and ' =====' in lines[i]:
                                model_header = lines[i].strip()
                                model_match = PATTERNS['model_header'].search(model_header)
                                if model_match:
                                    model_name = model_match.group(1).strip()
                                    
                                    # Initialize model data
                                    model_data = {
                                        'model': model_name,
                                        'threads': None,
                                        'aggregator': None,
                                        'footprint': None,
                                        'cost': None
                                    }
                                    
                                    # Parse model details
                                    j = i + 1
                                    while j < len(lines) and not ('===== ' in lines[j] and ' =====' in lines[j]) and not '[TRACING] Exiting function:' in lines[j]:
                                        model_line = lines[j].strip()
                                        
                                        # Extract thread count
                                        if 'Threads:' in model_line:
                                            threads_match = PATTERNS['threads'].search(model_line)
                                            if threads_match:
                                                model_data['threads'] = int(threads_match.group(1))
                                        
                                        # Extract aggregator
                                        elif 'Stream aggregator:' in model_line:
                                            agg_match = PATTERNS['aggregator'].search(model_line)
                                            if agg_match:
                                                model_data['aggregator'] = agg_match.group(1)
                                        
                                        # Extract memory footprint
                                        elif 'Memory footprint:' in model_line and not model_line.startswith('Algorithm parameters:'):
                                            fp_match = PATTERNS['footprint'].search(model_line)
                                            if fp_match:
                                                model_data['footprint'] = fp_match.group(1)
                                        
                                        # Extract cost
                                        elif 'Total cost:' in model_line:
                                            cost_match = PATTERNS['model_cost'].search(model_line)
                                            if cost_match:
                                                model_data['cost'] = float(cost_match.group(1))
                                        
                                        j += 1
                                    
                                    # Add model to the call data
                                    call_data['models'].append(model_data)
                                    
                                    i = j - 1  # Move i to the line before the next model or exit
                            
                            # Look for execution time 
                            elif '[TRACING] Exiting function:' in lines[i]:
                                exit_match = PATTERNS['exit'].search(lines[i])
                                if exit_match and exit_match.group(1) == function_name:
                                    exec_time_us = int(exit_match.group(2))
                                    call_data['execution_time'] = exec_time_us / 1e6
                                
                                # Add the function call data to the results
                                function_data[base_function].append(call_data)
                                
                                # Move to the next function
                                break
                            
                            i += 1
            
            i += 1
    
    except FileNotFoundError:
        print(f"Error: File not found: {log_file_path}")
    except Exception as e:
        print(f"Error parsing log file: {e}")
    
    return function_data

def analyze_function(function_name: str, log_file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Analyze function calls for a specific function name.
    
    Args:
        function_name: Name of the function to analyze
        log_file_path: Path to the log file
        
    Returns:
        Dictionary with argument types as keys and function statistics
    """
    data = parse_log_file(log_file_path)
    
    if function_name not in data:
        return {}
    
    analysis = defaultdict(lambda: {
        "count": 0, 
        "costs": [], 
        "execution_times": [], 
        "all_models": []
    })
    
    for call in data[function_name]:
        args = call['args']
        cost = call['cost']
        execution_time = call['execution_time']
        models = call.get('models', [])
        
        analysis[args]["count"] += 1
        analysis[args]["costs"].append(cost)
        analysis[args]["execution_times"].append(execution_time)
        
        # Store all models
        for model in models:
            analysis[args]["all_models"].append(model)
    
    return analysis

def create_solver_iteration_function(log_file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Create a synthetic function that represents one complete solver iteration by aggregating
    all functions that are part of the iteration.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        Dictionary with synthetic function data
    """
    # Get all function data
    function_data = parse_log_file(log_file_path)
    
    # Get number of CG iterations
    num_iterations = extract_cg_iterations(log_file_path)
    
    # Initialize the synthetic function
    synthetic_function = {}
    
    # Track the maximum memory footprint
    max_memory_footprint = 0
    max_memory_footprint_str = "0 B"
    
    # Track functions that are part of iterations
    iteration_functions = []
    
    # Track model information
    model_info = {}
    
    # Identify functions that are part of iterations and calculate their contribution
    for function_name, function_calls in function_data.items():
        # Extract mxv argument types if this is the mxv function
        if function_name == 'mxv' and function_calls:
            # Get the first call's args as representative
            mxv_args = function_calls[0]['args']
        
        # Group calls by argument type
        calls_by_args = defaultdict(list)
        for call in function_calls:
            calls_by_args[call['args']].append(call)
        
        # Process each argument type
        for args, calls in calls_by_args.items():
            count = len(calls)
            per_iter_count = determine_per_iteration_count(count, num_iterations)
            
            if per_iter_count is not None and per_iter_count > 0:
                # This function is part of the iteration
                execution_times = [c.get('execution_time', 0.0) for c in calls if c.get('execution_time') is not None]
                
                if not execution_times:
                    continue
                
                # Calculate statistics for this function
                min_time, max_time, avg_time, std_dev = calculate_statistics(execution_times)
                
                # Store function info
                func_info = {
                    'function_name': function_name,
                    'args': args,
                    'per_iter_count': per_iter_count,
                    'total_count': count,
                    'avg_exec_time': avg_time,
                    'min_exec_time': min_time,
                    'max_exec_time': max_time,
                    'per_iter_exec_time': avg_time * per_iter_count,  # Time contribution to each iteration
                    'calls': calls
                }
                
                iteration_functions.append(func_info)
                
                # Update max memory footprint
                for call in calls:
                    for model in call.get('models', []):
                        footprint_str = model.get('footprint', '0 B')
                        footprint_bytes = parse_memory_footprint(footprint_str)
                        if footprint_bytes > max_memory_footprint:
                            max_memory_footprint = footprint_bytes
                            max_memory_footprint_str = footprint_str
                
                # Collect model info for cost calculation
                for call in calls:
                    for model in call.get('models', []):
                        model_key = (
                            model.get('model', '') + "-additive",  # Rename model
                            model.get('aggregator', ''),
                            model.get('threads', 0)
                        )
                        
                        # Store model information if not already stored
                        if model_key not in model_info:
                            model_info[model_key] = {
                                'model': model.get('model', '') + "-additive",
                                'threads': model.get('threads', 0),
                                'aggregator': model.get('aggregator', ''),
                                'level_range': model.get('level_range'),
                                'cost': 0.0
                            }
                        
                        # Calculate average cost per call for this model
                        if model.get('cost') is not None and count > 0:
                            cost_per_call = model.get('cost', 0.0)
                            # Total contribution to iterations
                            cost_contribution = cost_per_call * per_iter_count
                            model_info[model_key]['cost'] += cost_contribution
    
    # Calculate per-iteration totals
    if iteration_functions:
        # Calculate total execution time per iteration (sum of per_iter_exec_time for each function)
        total_exec_time_per_iter = sum(func['per_iter_exec_time'] for func in iteration_functions)
        
        # Calculate min/max range using the proportions of min/max to avg for each function
        min_factor = sum(func['min_exec_time'] * func['per_iter_count'] for func in iteration_functions) / \
                    sum(func['avg_exec_time'] * func['per_iter_count'] for func in iteration_functions) \
                    if sum(func['avg_exec_time'] * func['per_iter_count'] for func in iteration_functions) > 0 else 1.0
                    
        max_factor = sum(func['max_exec_time'] * func['per_iter_count'] for func in iteration_functions) / \
                    sum(func['avg_exec_time'] * func['per_iter_count'] for func in iteration_functions) \
                    if sum(func['avg_exec_time'] * func['per_iter_count'] for func in iteration_functions) > 0 else 1.0
        
        min_exec_time_per_iter = total_exec_time_per_iter * min_factor
        max_exec_time_per_iter = total_exec_time_per_iter * max_factor
        
        # Create synthetic execution times list - one entry per iteration
        # All iterations get the same calculated values since we're using averages
        exec_times = [total_exec_time_per_iter] * num_iterations
        
        # Create the synthetic function entry with mxv args if available
        aggregated_key = "aggregated"
        if mxv_args:
            aggregated_key = f"aggregated for mxv - {mxv_args}"
        
        synthetic_function[aggregated_key] = {
            "count": num_iterations,
            "per_iter_count": 1,
            "costs": [sum(model_info[k]['cost'] for k in model_info) / num_iterations],  # Average cost per iteration
            "execution_times": exec_times,
            "execution_time_min": min_exec_time_per_iter,
            "execution_time_max": max_exec_time_per_iter,
            "execution_time_avg": total_exec_time_per_iter,
            "execution_time_stddev": 0.0,  # Standard deviation is 0 since all values are the same
            "all_models": []
        }
        
        # Add model data
        for model_key, model_data in model_info.items():
            # Adjust cost to be average per iteration
            model_data['cost'] = model_data['cost'] / num_iterations
            model_data['footprint'] = max_memory_footprint_str
            synthetic_function[aggregated_key]["all_models"].append(model_data)
    
    return synthetic_function

def analyze_all_functions(log_file_path: str) -> None:
    """
    Analyze all functions found in the given log file.
    
    Args:
        log_file_path: Path to the log file
    """
    function_data = parse_log_file(log_file_path)
    
    if not function_data:
        print(f"No function calls found in log file: {log_file_path}")
        return
    
    # Create a synthetic "Solver iteration" function
    synthetic_function = create_solver_iteration_function(log_file_path)
    
    # Get number of CG iterations
    num_iterations = extract_cg_iterations(log_file_path)
    
    # Print summary of functions found
    print(f"Found {len(function_data) + 1} distinct functions in log file (including synthetic function)")
    print("=" * 80)
    
    # First print the synthetic function if it has data
    if synthetic_function:
        print("\nAnalysis for function 'Solver_iteration':")
        print("=" * 80)
        
        for args, data in synthetic_function.items():
            print(f"Argument types: {args}")
            print(f"Invocation count: {num_iterations} (per solver iteration: 1)")
            
            # Print all models
            for model_data in data.get('all_models', []):
                print(f"Model: {model_data.get('model', 'Unknown')}")
                print(f"Threads: {model_data.get('threads', 'Unknown')}")
                print(f"Stream aggregator: {model_data.get('aggregator', 'Unknown')}")
                print(f"Memory footprint: {model_data.get('footprint', 'Unknown')}")
                
                if model_data.get('level_range'):
                    print(f"Level range: {model_data.get('level_range')}")
                    
                print(f"Predicted cost: {model_data.get('cost', 0.0):.5e}")
                print("-" * 80)
            
            # Process execution times with statistics
            if data.get('execution_time_min') is not None:
                min_time = data.get('execution_time_min')
                max_time = data.get('execution_time_max')
                avg_time = data.get('execution_time_avg')
                std_dev = data.get('execution_time_stddev')
                
                if std_dev is not None:
                    print(f"Execution time (seconds): min={min_time:.5e}, max={max_time:.5e}, avg={avg_time:.5e}, std_dev={std_dev:.5e}")
                else:
                    print(f"Execution time (seconds): min={min_time:.5e}, max={max_time:.5e}, avg={avg_time:.5e}")
            
            print("-" * 80)
    
    # Analyze each regular function
    for function_name in sorted(function_data.keys()):
        print("\n")  # Add space between function analyses
        print_function_analysis(function_name, log_file_path)

def calculate_statistics(values: List[float]) -> Tuple[float, float, float, Optional[float]]:
    """Calculate min, max, avg, and std_dev for a list of values."""
    if not values:
        return 0.0, 0.0, 0.0, None
        
    min_val = min(values)
    max_val = max(values)
    avg_val = sum(values) / len(values)
    
    std_dev = None
    if len(values) > 1:
        std_dev = np.std(values)
        
    return min_val, max_val, avg_val, std_dev

def determine_per_iteration_count(count: int, num_iterations: int) -> Optional[int]:
    """
    Determine if a function is part of the iterative loop and how many times it's called per iteration.
    
    Args:
        count: Total invocation count
        num_iterations: Number of CG iterations
        
    Returns:
        Per-iteration count if it's part of the iterative loop, None otherwise
    """
    # If count is divisible by num_iterations, it's likely called exactly that many times per iteration
    if count % num_iterations == 0:
        return count // num_iterations
    
    # If count is divisible by (num_iterations - 1), it might be called in all but the last iteration
    if count % (num_iterations - 1) == 0 and num_iterations > 1:
        return count // (num_iterations - 1)
    
    # If count is slightly more than a multiple of num_iterations (e.g., setup + per iteration)
    for offset in range(1, 4):  # Try a few offsets
        if (count - offset) > 0 and (count - offset) % num_iterations == 0:
            return (count - offset) // num_iterations
    
    # If count is close to a multiple of num_iterations
    for per_iter in range(1, 20):  # Try reasonable per-iteration counts
        expected = per_iter * num_iterations
        if abs(count - expected) <= 2:  # Allow small deviation
            return per_iter
    
    # Not part of the iterative loop or doesn't fit a simple pattern
    return None

def print_function_analysis(function_name: str, log_file_path: str) -> None:
    """
    Print analysis of function calls for a specific function name.
    
    Args:
        function_name: Name of the function to analyze
        log_file_path: Path to the log file
    """
    analysis = analyze_function(function_name, log_file_path)
    
    if not analysis:
        print(f"No data found for function '{function_name}'")
        return
    
    # Get number of CG iterations
    num_iterations = extract_cg_iterations(log_file_path)
    
    print(f"Analysis for function '{function_name}':")
    print("=" * 80)
    
    for args, data in analysis.items():
        invocation_count = data['count']
        per_iteration_count = determine_per_iteration_count(invocation_count, num_iterations)
        
        print(f"Argument types: {args}")
        
        # Always include per-iteration information, using 0 if not part of iteration
        if per_iteration_count is not None:
            print(f"Invocation count: {invocation_count} (per solver iteration: {per_iteration_count})")
        else:
            print(f"Invocation count: {invocation_count} (per solver iteration: 0)")
        
        # Print all models
        models_seen = set()
        for model_data in data.get('all_models', []):
            # Create a key to avoid duplicates
            model_key = (
                model_data.get('model', ''),
                model_data.get('aggregator', ''),
                model_data.get('threads', 0)
            )
            
            if model_key in models_seen:
                continue
                
            models_seen.add(model_key)
            
            print(f"Model: {model_data.get('model', 'Unknown')}")
            print(f"Threads: {model_data.get('threads', 'Unknown')}")
            print(f"Stream aggregator: {model_data.get('aggregator', 'Unknown')}")
            print(f"Memory footprint: {model_data.get('footprint', 'Unknown')}")
            
            if model_data.get('level_range'):
                print(f"Level range: {model_data.get('level_range')}")
                
            # Use model-specific cost
            if model_data.get('cost') is not None:
                print(f"Predicted cost: {model_data.get('cost'):.5e}")
            else:
                print(f"Predicted cost: {data['costs'][0]:.5e}")
                
            print("-" * 80)
        
        # Process execution times
        execution_times = data['execution_times']
        if execution_times:
            min_time, max_time, avg_time, std_dev = calculate_statistics(execution_times)
            
            if len(execution_times) > 1:
                print(f"Execution time (seconds): min={min_time:.5e}, max={max_time:.5e}, avg={avg_time:.5e}, std_dev={std_dev:.5e}")
            else:
                print(f"Execution time (seconds): {execution_times[0]:.5e}")
        
        print("-" * 80)

def analyze_all_functions(log_file_path: str) -> None:
    """
    Analyze all functions found in the given log file.
    
    Args:
        log_file_path: Path to the log file
    """
    function_data = parse_log_file(log_file_path)
    
    if not function_data:
        print(f"No function calls found in log file: {log_file_path}")
        return
    
    # Create a synthetic "Solver iteration" function
    synthetic_function = create_solver_iteration_function(log_file_path)
    
    # Get number of CG iterations
    num_iterations = extract_cg_iterations(log_file_path)
    
    # Print summary of functions found
    print(f"Found {len(function_data) + 1} distinct functions in log file (including synthetic function)")
    print("=" * 80)
    
    # First print the synthetic function if it has data
    if synthetic_function:
        print("\nAnalysis for function 'Solver_iteration':")
        print("=" * 80)
        
        for args, data in synthetic_function.items():
            print(f"Argument types: {args}")
            print(f"Invocation count: {num_iterations} (per solver iteration: 1)")
            
            # Print all models
            for model_data in data.get('all_models', []):
                print(f"Model: {model_data.get('model', 'Unknown')}")
                print(f"Threads: {model_data.get('threads', 'Unknown')}")
                print(f"Stream aggregator: {model_data.get('aggregator', 'Unknown')}")
                print(f"Memory footprint: {model_data.get('footprint', 'Unknown')}")
                
                if model_data.get('level_range'):
                    print(f"Level range: {model_data.get('level_range')}")
                    
                print(f"Predicted cost: {model_data.get('cost', 0.0):.5e}")
                print("-" * 80)
            
            # Process execution times with statistics
            if data.get('execution_time_min') is not None:
                min_time = data.get('execution_time_min')
                max_time = data.get('execution_time_max')
                avg_time = data.get('execution_time_avg')
                std_dev = data.get('execution_time_stddev')
                
                if std_dev is not None:
                    print(f"Execution time (seconds): min={min_time:.5e}, max={max_time:.5e}, avg={avg_time:.5e}, std_dev={std_dev:.5e}")
                else:
                    print(f"Execution time (seconds): min={min_time:.5e}, max={max_time:.5e}, avg={avg_time:.5e}")
            
            print("-" * 80)
    
    # Analyze each regular function
    for function_name in sorted(function_data.keys()):
        print("\n")  # Add space between function analyses
        print_function_analysis(function_name, log_file_path)

def parse_memory_footprint(footprint_str: str) -> int:
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
        return int(value * 1024)
    elif unit == 'MB':
        return int(value * 1024 * 1024)
    elif unit == 'GB':
        return int(value * 1024 * 1024 * 1024)
    elif unit == 'TB':
        return int(value * 1024 * 1024 * 1024 * 1024)
    elif unit == 'PB':
        return int(value * 1024 * 1024 * 1024 * 1024 * 1024)
    else:  # Bytes
        return int(value)

def main() -> None:
    """Main function to process command-line arguments and analyze the log file."""
    if len(sys.argv) < 2:
        print("Usage: python cost_and_time_parser.py <log_file_path>")
        return
    
    log_file_path = sys.argv[1]
    analyze_all_functions(log_file_path)

if __name__ == "__main__":
    main()