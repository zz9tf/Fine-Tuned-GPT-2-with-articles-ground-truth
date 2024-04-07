import subprocess

# Execute proxy command
command = "source /etc/network_turbo"

# Execute the command
output = subprocess.run(command, shell=True, capture_output=True, text=True)

# Print the output
print(output.stdout)