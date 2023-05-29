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
    sbatch ./code/exe.sh
fi
