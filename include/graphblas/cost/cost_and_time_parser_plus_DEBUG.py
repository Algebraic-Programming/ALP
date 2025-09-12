import re
from collections import defaultdict
import sys
import numpy as np

def parse_log_file(log_file_path):
    """
    Parse the log file and extract function calls with their argument types, costs,
    and execution times.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Dictionary with function calls information
    """
    function_data = defaultdict(list)
    current_function = None
    current_args = None
    current_threads = None
    current_aggregator = None
    current_model = None
    current_footprint = None
    current_levels = []
    
    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Match function entry
            entering_match = re.search(r'\[TRACING\] Entering function: (\w+<[^>]*>|\w+<\d+>|\w+) with (\d+) arguments', line)
            if entering_match:
                current_function = entering_match.group(1)
                current_threads = None
                current_aggregator = None
                current_model = None
                current_footprint = None
                current_levels = []
                continue
                
            # Match argument types
            arg_types_match = re.search(r'\[TRACING\] Argument types: (.*)', line)
            if arg_types_match and current_function:
                current_args = arg_types_match.group(1).strip()
                continue
            
            # Match model name
            model_match = re.search(r'===== (.*) Kernel Cost Prediction =====', line)
            if model_match and current_function:
                current_model = model_match.group(1).strip()
                continue
                
            # Match threads
            threads_match = re.search(r'Threads: (\d+)', line)
            if threads_match and current_function:
                current_threads = int(threads_match.group(1))
                continue
                
            # Match stream aggregator
            aggregator_match = re.search(r'Stream aggregator: (\w+)', line)
            if aggregator_match and current_function:
                current_aggregator = aggregator_match.group(1).strip()
                continue
                
            # Match memory footprint
            footprint_match = re.search(r'Memory footprint: ([\d.]+ [KMGTP]?B)', line)
            if footprint_match and current_function:
                current_footprint = footprint_match.group(1).strip()
                continue
                
            # Match level information in computation breakdown
            level_match = re.search(r'Superstep type \d+ \(level (\d+)\):', line)
            if level_match and current_function:
                current_levels.append(int(level_match.group(1)))
                continue
                
            # Match predicted cost
            cost_match = re.search(r'\[TRACING\] Predicted cost: ([0-9.e+-]+)', line)
            if cost_match and current_function and current_args:
                cost = float(cost_match.group(1))
                # Keep track of cost but don't create a record yet, wait for execution time
                continue
                
            # Match execution time (when exiting function)
            exit_match = re.search(r'\[TRACING\] Exiting function: (\w+<[^>]*>|\w+<\d+>|\w+) \(took (\d+)μs\)', line)
            if exit_match and current_function and current_args and exit_match.group(1) == current_function:
                execution_time_us = int(exit_match.group(2))
                execution_time_s = execution_time_us / 1e6  # Convert μs to seconds
                
                # Extract base function name (remove template part if present)
                base_function = re.match(r'(\w+)', current_function).group(1)
                
                # Calculate min and max levels
                level_range = None
                if current_levels:
                    level_min = min(current_levels)
                    level_max = max(current_levels)
                    level_range = [level_min, level_max]
                
                function_data[base_function].append({
                    'full_name': current_function,
                    'args': current_args,
                    'cost': cost,
                    'execution_time': execution_time_s,
                    'threads': current_threads,
                    'aggregator': current_aggregator,
                    'model': current_model,
                    'footprint': current_footprint,
                    'level_range': level_range
                })
                current_args = None
                
    return function_data

def extract_iterations(log_file_path):
    """
    Extract the number of iterations from the log file.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        int: Number of iterations, or None if not found
    """
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(r'Benchmark completed successfully and took (\d+) iterations', line)
            if match:
                return int(match.group(1))
    return None

def analyze_function(function_name, log_file_path):
    """
    Analyze function calls for a specific function name.
    
    Args:
        function_name (str): Name of the function to analyze
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Dictionary with argument types as keys and function statistics
    """
    data = parse_log_file(log_file_path)
    
    if function_name not in data:
        return {}
    
    analysis = defaultdict(lambda: {
        "count": 0, 
        "costs": [], 
        "execution_times": [], 
        "details": []
    })
    
    for call in data[function_name]:
        args = call['args']
        cost = call['cost']
        execution_time = call['execution_time']
        
        analysis[args]["count"] += 1
        analysis[args]["costs"].append(cost)
        analysis[args]["execution_times"].append(execution_time)
        analysis[args]["details"].append({
            'threads': call.get('threads'),
            'aggregator': call.get('aggregator'),
            'model': call.get('model'),
            'footprint': call.get('footprint'),
            'level_range': call.get('level_range')
        })
    
    return analysis

def print_function_analysis(function_name, log_file_path):
    """
    Print analysis of function calls for a specific function name.
    For each parameter group, verify that all cost predictions are consistent.
    
    Args:
        function_name (str): Name of the function to analyze
        log_file_path (str): Path to the log file
    """
    analysis = analyze_function(function_name, log_file_path)
    
    if not analysis:
        print(f"No data found for function '{function_name}'")
        return
    
    print(f"Analysis for function '{function_name}':")
    print("=" * 80)
    
    for args, data in analysis.items():
        print(f"Argument types: {args}")
        print(f"Invocation count: {data['count']}")
        
        # Print the first detail record (assuming they're all the same for the same args)
        if data['details'] and data['details'][0]:
            detail = data['details'][0]
            if detail['model']:
                print(f"Model: {detail['model']}")
            if detail['threads']:
                print(f"Threads: {detail['threads']}")
            if detail['aggregator']:
                print(f"Stream aggregator: {detail['aggregator']}")
            if detail['footprint']:
                print(f"Memory footprint: {detail['footprint']}")
            if detail['level_range']:
                print(f"Level range: {detail['level_range']}")
        
        # Check if all costs are consistent
        costs = data['costs']
        if len(costs) > 1:
            # Use a small epsilon for floating point comparison
            epsilon = 1e-10
            reference_cost = costs[0]
            
            # Check if all costs are the same (within epsilon)
            consistent = all(abs(cost - reference_cost) < epsilon for cost in costs)
            
            if not consistent:
                print("ERROR: Inconsistent cost predictions for the same parameter set:")
                for i, cost in enumerate(costs):
                    print(f"  Prediction {i+1}: {cost}")
                print("-" * 80)
                continue
        
        # If we get here, all costs are consistent or there's just one cost
        print(f"Predicted cost: {costs[0]}")
        
        # Process execution times
        execution_times = data['execution_times']
        if execution_times:
            min_time = min(execution_times)
            max_time = max(execution_times)
            avg_time = sum(execution_times) / len(execution_times)
            
            # Calculate standard deviation if we have more than one sample
            if len(execution_times) > 1:
                std_dev = np.std(execution_times)
                print(f"Execution time (seconds): min={min_time:.5e}, max={max_time:.5e}, avg={avg_time:.5e}, std_dev={std_dev:.5e}")
            else:
                print(f"Execution time (seconds): {execution_times[0]:.5e}")
        
        print("-" * 80)

def analyze_all_functions(log_file_path, num_itters=None):
    """
    Analyze all functions found in the given log file.
    
    Args:
        log_file_path (str): Path to the log file
        num_itters (int, optional): Number of iterations to use for separating
                                  preprocessing and iterative operations
    """
    # Parse the log file to get all function data
    function_data = parse_log_file(log_file_path)
    
    if not function_data:
        print(f"No function calls found in log file: {log_file_path}")
        return
    
    # Print summary of functions found
    print(f"Found {len(function_data)} distinct functions in log file")
    print("=" * 80)
    
    if num_itters:
        analyze_with_iteration_separation(function_data, log_file_path, num_itters)
    else:
        # Standard analysis without iteration separation
        for function_name in sorted(function_data.keys()):
            # Create space between function analyses
            print("\n")
            print_function_analysis(function_name, log_file_path)

def analyze_with_iteration_separation(function_data, log_file_path, num_itters):
    """
    Analyze functions by separating them into preprocessing and iterative categories.
    
    Args:
        function_data (dict): Dictionary with function call data
        log_file_path (str): Path to the log file
        num_itters (int): Number of iterations to use for separation
    """
    preprocessing_functions = {}
    iterative_functions = {}
    
    # Categorize functions as preprocessing or iterative
    for function_name, calls in function_data.items():
        # Group by argument types first
        args_groups = defaultdict(list)
        for call in calls:
            args_groups[call['args']].append(call)
        
        # Process each argument group
        for args, arg_calls in args_groups.items():
            call_count = len(arg_calls)
            
            # Consider a function iterative if it's called at least num_itters-1 times
            # This handles cases where a function might be called slightly fewer times due to end conditions
            if call_count >= num_itters - 1:
                # Calculate the number of complete iterations and any deficit
                iterations = call_count // num_itters
                remainder = call_count % num_itters
                
                # If remainder is close to num_itters, it's likely another iteration with a deficit
                if remainder >= num_itters - 1:
                    iterations += 1
                    deficit = num_itters - remainder
                else:
                    deficit = 0
                
                if function_name not in iterative_functions:
                    iterative_functions[function_name] = {}
                
                # Add to iterative functions with deficit information
                iterative_functions[function_name][args] = {
                    "count": call_count,
                    "iterations": iterations,
                    "deficit": deficit,
                    "costs": [arg_calls[0]['cost']],  # Assuming costs are consistent
                    "execution_times": [call['execution_time'] for call in arg_calls],  # All execution times
                    "details": arg_calls[0]  # Store the first call's details for display
                }
                
                # Any small remainder (less than num_itters-1) goes to preprocessing
                if remainder > 0 and remainder < num_itters - 1:
                    if function_name not in preprocessing_functions:
                        preprocessing_functions[function_name] = {}
                    
                    # Add the first 'remainder' calls to preprocessing
                    preprocessing_functions[function_name][args] = {
                        "count": remainder,
                        "costs": [arg_calls[0]['cost']],  # Assuming costs are consistent
                        "execution_times": [arg_calls[i]['execution_time'] for i in range(remainder)],
                        "details": arg_calls[0]  # Store the first call's details for display
                    }
            else:
                # This function is called less than num_itters-1 times, so it's preprocessing
                if function_name not in preprocessing_functions:
                    preprocessing_functions[function_name] = {}
                
                preprocessing_functions[function_name][args] = {
                    "count": call_count,
                    "costs": [arg_calls[0]['cost']],  # Assuming costs are consistent
                    "execution_times": [call['execution_time'] for call in arg_calls],
                    "details": arg_calls[0]  # Store the first call's details for display
                }
    
    # Print preprocessing functions
    print("\n\nPREPROCESSING FUNCTIONS")
    print("=" * 80)
    
    if not preprocessing_functions:
        print("No preprocessing functions found.")
    else:
        for function_name in sorted(preprocessing_functions.keys()):
            print(f"\nAnalysis for function '{function_name}':")
            print("=" * 80)
            
            for args, data in preprocessing_functions[function_name].items():
                print(f"Argument types: {args}")
                print(f"Invocation count: {data['count']}")
                
                # Print details
                if 'details' in data:
                    detail = data['details']
                    if detail.get('model'):
                        print(f"Model: {detail['model']}")
                    if detail.get('threads'):
                        print(f"Threads: {detail['threads']}")
                    if detail.get('aggregator'):
                        print(f"Stream aggregator: {detail['aggregator']}")
                    if detail.get('footprint'):
                        print(f"Memory footprint: {detail['footprint']}")
                    if detail.get('level_range'):
                        print(f"Level range: {detail['level_range']}")
                
                print(f"Predicted cost: {data['costs'][0]}")
                
                # Print execution time statistics
                if data['execution_times']:
                    execution_times = data['execution_times']
                    min_time = min(execution_times)
                    max_time = max(execution_times)
                    avg_time = sum(execution_times) / len(execution_times)
                    
                    if len(execution_times) > 1:
                        std_dev = np.std(execution_times)
                        print(f"Execution time (seconds): min={min_time:.5e}, max={max_time:.5e}, avg={avg_time:.5e}, std_dev={std_dev:.5e}")
                    else:
                        print(f"Execution time (seconds): {execution_times[0]:.5e}")
                
                print("-" * 80)
    
    # Print iterative functions
    print("\n\nITERATIVE FUNCTIONS")
    print("=" * 80)
    
    if not iterative_functions:
        print("No iterative functions found.")
    else:
        # Organize execution times by iteration for each function
        function_execution_times_by_iteration = {}
        
        # Initialize a data structure to hold execution times for each iteration
        iteration_execution_times = defaultdict(list)
        
        # Track the maximum number of iterations seen
        max_iterations = 0
        
        for function_name in sorted(iterative_functions.keys()):
            print(f"\nAnalysis for function '{function_name}':")
            print("=" * 80)
            
            for args, data in iterative_functions[function_name].items():
                # Format the display to show any deficit
                if data['deficit'] > 0:
                    invocation_display = f"{num_itters} × {data['iterations']} (-{data['deficit']})"
                else:
                    invocation_display = f"{num_itters} × {data['iterations']}"
                
                print(f"Argument types: {args}")
                print(f"Invocation count: {invocation_display}")
                
                # Print details
                if 'details' in data:
                    detail = data['details']
                    if detail.get('model'):
                        print(f"Model: {detail['model']}")
                    if detail.get('threads'):
                        print(f"Threads: {detail['threads']}")
                    if detail.get('aggregator'):
                        print(f"Stream aggregator: {detail['aggregator']}")
                    if detail.get('footprint'):
                        print(f"Memory footprint: {detail['footprint']}")
                    if detail.get('level_range'):
                        print(f"Level range: {detail['level_range']}")
                
                print(f"Predicted cost: {data['costs'][0]}")
                
                # Print execution time statistics for this function
                if data['execution_times']:
                    execution_times = data['execution_times']
                    min_time = min(execution_times)
                    max_time = max(execution_times)
                    avg_time = sum(execution_times) / len(execution_times)
                    
                    if len(execution_times) > 1:
                        std_dev = np.std(execution_times)
                        print(f"Execution time (seconds): min={min_time:.5e}, max={max_time:.5e}, avg={avg_time:.5e}, std_dev={std_dev:.5e}")
                    else:
                        print(f"Execution time (seconds): {execution_times[0]:.5e}")
                
                # Organize execution times by iteration
                function_key = f"{function_name}:{args}"
                times = data['execution_times']
                
                # Update the maximum iterations seen
                max_iterations = max(max_iterations, data['iterations'])
                
                # Distribute execution times to iterations
                # This assumes that functions are called in order across iterations
                for i in range(min(data['iterations'], len(times) // num_itters)):
                    # For each iteration, get the corresponding times for this function
                    start_idx = i * num_itters
                    end_idx = min(start_idx + num_itters, len(times))
                    iter_times = times[start_idx:end_idx]
                    
                    # Add these times to the corresponding iteration
                    iteration_execution_times[i].extend(iter_times)
                
                print("-" * 80)
        
        # Print the total execution time per iteration statistics
        print("\nTOTAL EXECUTION TIME PER ITERATION")
        print("=" * 80)
        
        # Calculate total execution time for each iteration
        if iteration_execution_times:
            # Sum the execution times for each iteration
            iteration_total_times = [sum(times) for iter_idx, times in sorted(iteration_execution_times.items())]
            
            if iteration_total_times:
                min_exec_time = min(iteration_total_times)
                max_exec_time = max(iteration_total_times)
                avg_exec_time = sum(iteration_total_times) / len(iteration_total_times)
                
                print(f"Total execution time for one iteration (seconds): {avg_exec_time:.5e}")
                
                if len(iteration_total_times) > 1:
                    std_dev_exec_time = np.std(iteration_total_times)
                    print(f"Execution time statistics: min={min_exec_time:.5e}, max={max_exec_time:.5e}, avg={avg_exec_time:.5e}, std_dev={std_dev_exec_time:.5e}")
                else:
                    print(f"Only one complete iteration found.")
        else:
            print("No complete iterations found for execution time analysis.")
        
        print("=" * 80)
        
        # Also show the predicted cost per iteration for comparison
        total_cost_per_iteration = sum(data['costs'][0] for func_data in iterative_functions.values() 
                                      for data in func_data.values())
        
        print("\nPREDICTED COST PER ITERATION (FOR REFERENCE)")
        print("=" * 80)
        print(f"Total predicted cost for one iteration: {total_cost_per_iteration:.5e}")
        print("=" * 80)

def main():
    """Example usage of the module"""
    
    if len(sys.argv) < 2:
        print("Usage: python cost_and_time_parser.py <log_file_path> [function_name]")
        return
    
    log_file_path = sys.argv[1]
    
    # Extract the number of iterations from the log file
    num_itters = extract_iterations(log_file_path)
    
    if num_itters is None:
        print("Warning: Could not extract the number of iterations from the log file.")
        print("The analysis will proceed without iteration separation.")
    else:
        print(f"Extracted {num_itters} iterations from the log file.")
    
    if len(sys.argv) >= 3:
        # Analyze a specific function
        function_name = sys.argv[2]
        print_function_analysis(function_name, log_file_path)
    else:
        # Analyze all functions
        analyze_all_functions(log_file_path, num_itters)

if __name__ == "__main__":
    main()