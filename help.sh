source ./code/read_env.sh

if [[ $1 == 'jobs' ]]; then
    squeue -u $( read_var USER )
elif [[ $1 == 'push' ]]; then
    scp .env $(read_var REMOTE_CONNECTION):$(read_var PATH)
elif [[ $1 == 'kill' ]]; then
    scancel $2
elif [[ $1 == 'info' ]]; then
    sinfo
else
    # Read env file
    source ./code/read_env.sh

    # Activate the virtual environment if needed
    source $( read_env PY_ENV ) gpt2

    # Path to your Python executable and script
    PYTHON_EXECUTABLE=python
    PYTHON_SCRIPT=--version

    # Run the Python code
    $PYTHON_EXECUTABLE $PYTHON_SCRIPT
    
    sbatch ./code/exe.sh
fi
