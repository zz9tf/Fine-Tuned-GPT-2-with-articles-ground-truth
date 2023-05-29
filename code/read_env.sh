read_env() {
  if [ -z "$1" ]; then
    echo "Please enter the environment variable to read!"
    return 1
  fi

  local ENV_FILE='../.env'
  if [ ! -z "$2" ]; then
    ENV_FILE="$2"
  fi

  local VAR=$(grep $1 "$ENV_FILE" | xargs)
  IFS="=" read -ra VAR <<< "$VAR"
  echo ${VAR[1]}
}

# echo 'load read_env'
