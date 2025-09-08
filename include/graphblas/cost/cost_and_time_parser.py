import re
from collections import defaultdict
import sys

def parse_log_file(log_file_path):
    """
    Parse the log file and extract function calls with their argument types and costs.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Dictionary with function calls information
    """
    function_data = defaultdict(list)
    current_function = None
    current_args = None
    
    with open(log_file_path, 'r') as file:
        for line in file:
            # Match function entry
            entering_match = re.search(r'\[TRACING\] Entering function: (\w+<[^>]*>|\w+<\d+>|\w+) with (\d+) arguments', line)
            if entering_match:
                current_function = entering_match.group(1)
                continue
                
            # Match argument types
            arg_types_match = re.search(r'\[TRACING\] Argument types: (.*)', line)
            if arg_types_match and current_function:
                current_args = arg_types_match.group(1).strip()
                continue
                
            # Match predicted cost
            cost_match = re.search(r'\[TRACING\] Predicted cost: ([0-9.e+-]+)', line)
            if cost_match and current_function and current_args:
                cost = float(cost_match.group(1))
                # Extract base function name (remove template part if present)
                base_function = re.match(r'(\w+)', current_function).group(1)
                function_data[base_function].append({
                    'full_name': current_function,
                    'args': current_args,
                    'cost': cost
                })
                current_args = None
                
    return function_data

def analyze_function(function_name, log_file_path):
    """
    Analyze function calls for a specific function name.
    
    Args:
        function_name (str): Name of the function to analyze
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Dictionary with argument types as keys and list of (count, cost) as values
    """
    data = parse_log_file(log_file_path)
    
    if function_name not in data:
        return {}
    
    analysis = defaultdict(lambda: {"count": 0, "costs": []})
    
    for call in data[function_name]:
        args = call['args']
        cost = call['cost']
        
        analysis[args]["count"] += 1
        analysis[args]["costs"].append(cost)
    
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
            
            # Check if this function with these args is called at least num_itters times
            if call_count >= num_itters:
                # Number of complete iterations
                iterations = call_count // num_itters
                
                # Remainder calls go to preprocessing
                remainder = call_count % num_itters
                
                if function_name not in iterative_functions:
                    iterative_functions[function_name] = {}
                
                # Add to iterative functions
                iterative_functions[function_name][args] = {
                    "count": iterations * num_itters,
                    "iterations": iterations,
                    "costs": [arg_calls[0]['cost']],  # Assuming costs are consistent
                    "calls_per_iteration": num_itters // num_itters  # This is just 1 for now
                }
                
                # If there are remainder calls, add them to preprocessing
                if remainder > 0:
                    if function_name not in preprocessing_functions:
                        preprocessing_functions[function_name] = {}
                    
                    preprocessing_functions[function_name][args] = {
                        "count": remainder,
                        "costs": [arg_calls[0]['cost']]  # Assuming costs are consistent
                    }
            else:
                # This function is called less than num_itters times, so it's preprocessing
                if function_name not in preprocessing_functions:
                    preprocessing_functions[function_name] = {}
                
                preprocessing_functions[function_name][args] = {
                    "count": call_count,
                    "costs": [arg_calls[0]['cost']]  # Assuming costs are consistent
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
                print(f"Predicted cost: {data['costs'][0]}")
                print("-" * 80)
    
    # Print iterative functions
    print("\n\nITERATIVE FUNCTIONS")
    print("=" * 80)
    
    if not iterative_functions:
        print("No iterative functions found.")
    else:
        # Calculate total cost per iteration
        total_cost_per_iteration = 0.0
        
        for function_name in sorted(iterative_functions.keys()):
            print(f"\nAnalysis for function '{function_name}':")
            print("=" * 80)
            
            for args, data in iterative_functions[function_name].items():
                print(f"Argument types: {args}")
                print(f"Invocation count: {num_itters} × {data['iterations']} times")
                print(f"Predicted cost: {data['costs'][0]}")
                print("-" * 80)
                
                # Add to the total cost per iteration
                # Each function contributes its cost times how many times it appears in one iteration
                total_cost_per_iteration += data['costs'][0]
        
        # Print the total cost of one iteration
        print("\nTOTAL COST PER ITERATION")
        print("=" * 80)
        print(f"Total cost for one iteration: {total_cost_per_iteration}")
        print(f"(Sum of the predicted costs of all functions executed in one iteration)")
        print("=" * 80)

def main():
    """Example usage of the module"""
    
    if len(sys.argv) < 2:
        print("Usage: python cost_and_time_parser.py <log_file_path> [function_name] [num_itters]")
        return
    
    log_file_path = sys.argv[1]
    
    if len(sys.argv) >= 4:
        # Analyze with iteration separation
        try:
            num_itters = int(sys.argv[3])
        except ValueError:
            print(f"Error: num_itters must be an integer, got '{sys.argv[3]}'")
            return
        
        if len(sys.argv) >= 3:
            # Analyze a specific function with iteration separation
            function_name = sys.argv[2]
            # For a specific function, we'll still use print_function_analysis
            # as the separation is most useful for the overall analysis
            print_function_analysis(function_name, log_file_path)
        else:
            # Analyze all functions with iteration separation
            analyze_all_functions(log_file_path, num_itters)
    elif len(sys.argv) >= 3:
        # Try to parse as num_itters first
        try:
            num_itters = int(sys.argv[2])
            # If successful, analyze all functions with iteration separation
            analyze_all_functions(log_file_path, num_itters)
        except ValueError:
            # Not a number, must be a function name
            function_name = sys.argv[2]
            print_function_analysis(function_name, log_file_path)
    else:
        # Standard analysis of all functions
        analyze_all_functions(log_file_path)

if __name__ == "__main__":
    main()