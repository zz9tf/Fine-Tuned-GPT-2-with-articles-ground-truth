import time

def evaluate_time(func):
    # Start the timer
    start_time = time.time()

    # The step you want to evaluate
    # For example:
    result = func()

    # End the timer
    end_time = time.time()

    # Calculate the time taken
    elapsed_time = end_time - start_time

    print(f'Time taken for the step: {elapsed_time} seconds')
    return result
