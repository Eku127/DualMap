import time
import logging
import csv
import numpy as np
from contextlib import contextmanager
from tabulate import tabulate
from pympler import asizeof


# Set up the module-level logger
logger = logging.getLogger(__name__)

"""
A context manager that measures the time taken for a block of code to execute.

Parameters:
description (str): A description of the block of code being measured. Default is an empty string.

Yields:
None

Returns:
None

Example:
>>> with measure_time_block("My code block"):
>>>     # Code to be measured
>>>     pass
My code block: 0.01 seconds
"""


@contextmanager
def measure_time_block(description=""):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        message = f"{description}: {elapsed_time:.4f} seconds"

        # Check if logger is configured
        if logging.getLogger().hasHandlers():
            logger.info(message)
        else:
           print(message)


def measure_time(func):
    """
    A decorator function that measures the execution time of another function.

    Parameters:
    func (function): The function to be measured.

    Returns:
    function: The decorated function that measures the execution time.

    Example:
    >>> @measure_time
    >>> def my_function():
    >>>     # Code to be measured
    >>>     pass
    >>> my_function()
    Done! Execution time of my_function function: 0.01 seconds
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # Call the function with any arguments it was called with
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds")
        return result  # Return the result of the function call
    return wrapper

@contextmanager
def timing_context(name, instance, results_attr_name='timing_results'):
    """
    Context manager for measuring execution time with optional storage.
    
    Args:
        name (str): Name of the block being measured.
        instance (object): Object to store timing results.
        results_attr_name (str): Name of the attribute to store timing results on the instance.
    """
    # initialize the dict
    if not hasattr(instance, results_attr_name):
        setattr(instance, results_attr_name, {})
    
    # get dict
    results_dict = getattr(instance, results_attr_name)
    
    # initialize the list with the given name
    if name not in results_dict:
        results_dict[name] = []

    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # store the elapsed time in the dict
    results_dict[name].append(elapsed_time)

def print_timing_results(label, timing_results):
    if not timing_results:
        return
    rows = []
    for key, times in timing_results.items():
        avg_time = np.mean(times)
        percentile_90 = np.percentile(times, 90)
        rows.append([key, f"{avg_time:.4f}", f"{percentile_90:.4f}"])
    logger.info(f"\n{label} Timing Results:")
    logger.info(tabulate(rows, headers=["Step", "Avg Time (s)", "90th Percentile (s)"], tablefmt="grid"))

def save_timing_results(timing_results, csv_file):
    if not timing_results:
        return
    
    rows = []
    for key, times in timing_results.items():
        avg_time = np.mean(times)
        percentile_90 = np.percentile(times, 90)
        rows.append([key, f"{avg_time:.4f}", f"{percentile_90:.4f}"])
    
    # Save results to CSV if a filename is provided
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Avg Time (s)", "90th Percentile (s)"])
        for row in rows:
            writer.writerow(row)
    logger.info(f"\nTiming results saved to {csv_file}")


# Utility function to get memory usage of local and global maps
def get_map_memory_usage(local_map, global_map):
    local_mb = asizeof.asizeof(local_map) / 1024 / 1024
    global_mb = asizeof.asizeof(global_map) / 1024 / 1024
    return {
        "local_map_mb": round(local_mb, 4),
        "global_map_mb": round(global_mb, 4)
    }
