import os
import argparse
from dotenv import load_dotenv, set_key, unset_key, dotenv_values

# Load .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

def show_all():
    """Show all variables in the .env file."""
    env_vars = dotenv_values(env_path)
    for key, value in env_vars.items():
        print(f'{key}={value}')

def set_variable(key, value):
    """Set or update a variable in the .env file."""
    set_key(env_path, key, value)
    print(f'{key} set to {value}')

def delete_variable(key):
    """Delete a variable from the .env file."""
    unset_key(env_path, key)
    print(f'{key} has been deleted')

def main():
    parser = argparse.ArgumentParser(description="Manage .env variables")
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for the "set" command
    parser_set = subparsers.add_parser('set', help='Set or update a variable')
    parser_set.add_argument('key', type=str, help='The key of the variable')
    parser_set.add_argument('value', type=str, help='The value of the variable')

    # Subparser for the "delete" command
    parser_delete = subparsers.add_parser('delete', help='Delete a variable')
    parser_delete.add_argument('key', type=str, help='The key of the variable')

    # Subparser for the "show" command
    parser_show = subparsers.add_parser('show', help='Show all variables')

    args = parser.parse_args()

    if args.command == 'set':
        set_variable(args.key, args.value)
    elif args.command == 'delete':
        delete_variable(args.key)
    elif args.command == 'show':
        show_all()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
