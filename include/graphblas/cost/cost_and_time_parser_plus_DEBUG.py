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
    'model': re.compile(r'===== (.*) Kernel Cost Prediction ====='),
    'threads': re.compile(r'Threads: (\d+)'),
    'aggregator': re.compile(r'Stream aggregator: (\w+)'),
    'footprint': re.compile(r'Memory footprint: ([\d.]+ [KMGTP]?B)'),
    'level': re.compile(r'Superstep type \d+ \(level (\d+)\):'),
    'cost': re.compile(r'\[TRACING\] Predicted cost: ([0-9.e+-]+)'),
    'exit': re.compile(r'\[TRACING\] Exiting function: (\w+<[^>]*>|\w+<\d+>|\w+) \(took (\d+)μs\)')
}

def extract_base_function_name(full_name: str) -> str:
    """Extract the base function name from a potentially templated function name."""
    match = re.match(r'(\w+)', full_name)
    if match:
        return match.group(1)
    return full_name

def parse_log_file(log_file_path: str) -> DefaultDict[str, List[Dict[str, Any]]]:
    """
    Parse the log file and extract function calls with their argument types, costs,
    and execution times.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        Dictionary with function calls information
    """
    function_data = defaultdict(list)
    current_function = None
    current_args = None
    current_cost = None
    current_threads = None
    current_aggregator = None
    current_model = None
    current_footprint = None
    current_levels = []
    
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Match function entry
                match = PATTERNS['function_entry'].search(line)
                if match:
                    current_function = match.group(1)
                    current_args = None
                    current_cost = None
                    current_threads = None
                    current_aggregator = None
                    current_model = None
                    current_footprint = None
                    current_levels = []
                    continue
                    
                # Only process other patterns if we're inside a function
                if not current_function:
                    continue
                
                # Match argument types
                match = PATTERNS['arg_types'].search(line)
                if match:
                    current_args = match.group(1).strip()
                    continue
                
                # Match model name
                match = PATTERNS['model'].search(line)
                if match:
                    current_model = match.group(1).strip()
                    continue
                    
                # Match threads
                match = PATTERNS['threads'].search(line)
                if match:
                    current_threads = int(match.group(1))
                    continue
                    
                # Match stream aggregator
                match = PATTERNS['aggregator'].search(line)
                if match:
                    current_aggregator = match.group(1).strip()
                    continue
                    
                # Match memory footprint
                match = PATTERNS['footprint'].search(line)
                if match:
                    current_footprint = match.group(1).strip()
                    continue
                    
                # Match level information
                match = PATTERNS['level'].search(line)
                if match:
                    current_levels.append(int(match.group(1)))
                    continue
                    
                # Match predicted cost
                match = PATTERNS['cost'].search(line)
                if match and current_args:
                    current_cost = float(match.group(1))
                    continue
                    
                # Match execution time (when exiting function)
                match = PATTERNS['exit'].search(line)
                if match and current_args and current_cost and match.group(1) == current_function:
                    execution_time_us = int(match.group(2))
                    execution_time_s = execution_time_us / 1e6  # Convert μs to seconds
                    
                    # Extract base function name
                    base_function = extract_base_function_name(current_function)
                    
                    # Calculate level range
                    level_range = None
                    if current_levels:
                        level_range = [min(current_levels), max(current_levels)]
                    
                    # Store the function call data
                    function_data[base_function].append({
                        'full_name': current_function,
                        'args': current_args,
                        'cost': current_cost,
                        'execution_time': execution_time_s,
                        'threads': current_threads,
                        'aggregator': current_aggregator,
                        'model': current_model,
                        'footprint': current_footprint,
                        'level_range': level_range
                    })
                    current_args = None  # Reset to prevent duplicate entries
    except FileNotFoundError:
        print(f"Error: File not found: {log_file_path}")
        return function_data
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return function_data
                
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
    
    print(f"Analysis for function '{function_name}':")
    print("=" * 80)
    
    for args, data in analysis.items():
        print(f"Argument types: {args}")
        print(f"Invocation count: {data['count']}")
        
        # Print the first detail record
        if data['details'] and data['details'][0]:
            detail = data['details'][0]
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
        
        # Check if all costs are consistent
        costs = data['costs']
        if costs:
            if len(costs) > 1:
                # Check for consistency
                epsilon = 1e-10
                reference_cost = costs[0]
                consistent = all(abs(cost - reference_cost) < epsilon for cost in costs)
                
                if not consistent:
                    print("ERROR: Inconsistent cost predictions for the same parameter set:")
                    for i, cost in enumerate(costs):
                        print(f"  Prediction {i+1}: {cost}")
                    print("-" * 80)
                    continue
            
            # If consistent or just one cost
            print(f"Predicted cost: {costs[0]}")
        
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
    
    # Print summary of functions found
    print(f"Found {len(function_data)} distinct functions in log file")
    print("=" * 80)
    
    # Analyze each function
    for function_name in sorted(function_data.keys()):
        print("\n")  # Add space between function analyses
        print_function_analysis(function_name, log_file_path)

def main() -> None:
    """Main function to process command-line arguments and analyze the log file."""
    if len(sys.argv) < 2:
        print("Usage: python cost_and_time_parser.py <log_file_path> [function_name]")
        return
    
    log_file_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        # Analyze a specific function
        function_name = sys.argv[2]
        print_function_analysis(function_name, log_file_path)
    else:
        # Analyze all functions
        analyze_all_functions(log_file_path)

if __name__ == "__main__":
    main()