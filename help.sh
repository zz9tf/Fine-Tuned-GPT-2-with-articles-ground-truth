source ./code/read_env.sh

if [[ $# -gt 0 ]]; then
    if [[ $1 == 'jobs' ]]; then
        squeue -u $(read_env USER)
    elif [[ $1 == 'push' ]]; then
        scp .env $(read_env REMOTE_CONNECTION):$(read_env PATH)
    elif [[ $1 == 'kill' ]]; then
        scancel $2
    elif [[ $1 == 'info' ]]; then
        sinfo
    elif [[ $1 == 'install' ]]; then
        source $( read_env PY_ENV ) gpt2
        pip install $2
    elif [[ $1 == 'build' && $2 == 'tensorflow' ]]; then
        docker stop $(docker ps -a -q)
        docker rm $(docker ps -a -q)
        docker volume rm $(docker volume ls -q)
        echo "y" | docker system prune -a
        echo "y" | docker builder prune --all
        cd custome-tensorflow && \
        docker build -t tensorflow-custom . && \
        docker run --name build-tensorflow -it tensorflow-custom

    else
        echo "Usage: source ./code/read_env.sh [command]"
        echo
        echo "Commands:"
        echo "  jobs                List jobs for the current user"
        echo "  push                Push .env file to a remote server"
        echo "  kill                Cancel a specific job"
        echo "  info                Display jobs information"
        echo "  build tensorflow    Build a custom tensorflow"
        echo
        echo "If no command is specified, a job will be submitted."
        echo
        echo "Additional options and functionality can be added as needed."
    fi
else
    rm -f *out
    # Activate the virtual environment if needed
    source $(read_env PY_ENV) gpt2
    python --version

    # Path to your Python executable and script
    sbatch ./code/exe.sh
fi
