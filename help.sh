source ./code/read_env.sh
if [[ $# -gt 0 ]]; then
    if [[ $1 == 'jobs' ]]; then
        squeue -u $( read_env USER )
    elif [[ $1 == 'push' ]]; then
        scp .env $(read_var REMOTE_CONNECTION):$(read_var PATH)
    elif [[ $1 == 'kill' ]]; then
        scancel $2
    elif [[ $1 == 'info' ]]; then
        sinfo
    else
        echo "Usage: source ./code/read_env.sh [command]"
        echo
        echo "Commands:"
        echo "  jobs          List jobs for the current user"
        echo "  push          Push .env file to a remote server"
        echo "  kill          Cancel a specific job"
        echo "  info          Display jobs information"
        echo
        echo "If no command is specified, a job will be submitted."
        echo
        echo "Additional options and functionality can be added as needed."
    fi
else
    # Activate the virtual environment if needed
    source $( read_env PY_ENV ) gpt2
    python --version

    # Path to your Python executable and script
    sbatch ./code/exe.sh
fi
