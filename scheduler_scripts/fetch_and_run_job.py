import requests
import time
from datetime import datetime
import subprocess
import os
import argparse

# mkdir -p ~/conda_setup
# mkdir -p ~/issaclab_setup
# ln -s ~/miniconda3 ~/conda_setup/miniconda3
# ln -s ~/Documents/issac_workspace/IsaacLabDRAIL ~/issaclab_setup/IsaacLabDRAIL

# /home/ayubi/Documents/issac_workspace/RoboQueue/submit_job_app/client_script

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Fetch and run jobs from web server')
    parser.add_argument('--miniconda-path', type=str, required=False,
                       help='Path to Miniconda installation', default="~/conda_setup/miniconda3")
    parser.add_argument('--repo-path', type=str, required=False,
                       help='Path to isaaclabDrail repository', default="~/issaclab_setup/IsaacLabDRAIL")
    parser.add_argument('--git-username', type=str, required=False,
                       help='Git username for repository access')
    parser.add_argument('--git-password', type=str, required=False,
                       help='Git password or personal access token')
    parser.add_argument('--enable-git-pull', action='store_true',
                       help='Enable automatic git pull before running jobs')
    return parser.parse_args()

# This file must be in the root dir of isaaclabDrail Repository.
# DRAIL_REPO_PATH = "./"
# MINICONDA_PATH = "/home/ayubi/miniconda3"
web_server_url = "https://drail.ngrok.dev"

def fetch_job():
    try:
        response = requests.get(web_server_url + '/api/jobs')
        if response.status_code == 200:
            job = response.json()
            if job:
                print(f"Found pending job")
            return job
        else:
            print(f"Failed to fetch jobs. Status code: {response.status_code}")
            print(f"Response content: {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching jobs: {e}")
        return []

def run_command(command):
    """
    Create and execute a bash script for the command.
    
    Args:
        command (str): Command to execute
        
    Returns:
        bool: True if command executed successfully, False otherwise
    """
    try:
        # Create a temporary bash script
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = "job_logs"
        script_path = f"{log_dir}/job_script_{timestamp}.sh"

        print("At command: ", command)
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        print("Created logs directory: ", log_dir)
        
        # Write the command to a bash script
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("echo 'Starting job'\n")
            f.write(f"cd {DRAIL_REPO_PATH}\n")
            f.write("echo 'Current directory:'\n")
            f.write("pwd\n")
            f.write(f"source {MINICONDA_PATH}/etc/profile.d/conda.sh\n")
            f.write(f"conda activate isaaclab_drail\n")
            f.write(f"{command}\n")

        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Execute the script and redirect output to log files
        stdout_file = f"{log_dir}/job_{timestamp}.out"
        stderr_file = f"{log_dir}/job_{timestamp}.err"
        
        result = subprocess.run(
            f"./{script_path}",
            shell=True,
            check=True,
            stdout=open(stdout_file, 'w'),
            stderr=open(stderr_file, 'w')
        )
        
        # Read and print the output files
        with open(stdout_file, 'r') as f:
            print(f"Command output:\n{f.read()}")
        
        with open(stderr_file, 'r') as f:
            stderr_content = f.read()
            if stderr_content:
                print(f"Command errors:\n{stderr_content}")
        
        # Clean up the script file (keep the logs)
        os.remove(script_path)
        return True
            
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed")
        return False
    except Exception as e:
        print(f"Error running command: {str(e)}")
        return False

def update_job_status(job_id, status):
    print(f"Updating job {job_id} status to {status}")
    response = requests.put(
        web_server_url + f'/api/jobs/{job_id}',
        json={'status': status}
    )
    if response.status_code != 200:
        print(f"Failed to update job {job_id} status to {status}")
        return False
    print(f"Job {job_id} status updated to {status}")
    return True

def git_pull():
    """
    Perform git pull in the repository.
    
    Returns:
        bool: True if git pull was successful, False otherwise
    """
    try:
        result = subprocess.run(
            ['git', 'pull'],
            check=True,
            capture_output=True,
            text=True,
            cwd=DRAIL_REPO_PATH
        )
        print(f"Git pull output:\n{result.stdout}")
        
        if result.stderr:
            print(f"Git pull warnings:\n{result.stderr}")
            
        return True
            
    except subprocess.CalledProcessError as e:
        print(f"Git pull failed:\n{e.stderr}")
        return False

def git_setup(username, password):
    """
    Set up Git credentials using provided username and password.
    
    Args:
        username (str): Git username
        password (str): Git password or personal access token
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        if not username or not password:
            print("Git credentials not provided")
            return False
            
        # Configure Git credentials
        subprocess.run(
            ['git', 'config', '--global', 'credential.helper', 'store'],
            check=True,
            capture_output=True,
            text=True,
            cwd=DRAIL_REPO_PATH
        )
        
        # Store credentials
        credential_input = f"url=https://github.com\nusername={username}\npassword={password}\n"
        subprocess.run(
            ['git', 'credential', 'approve'],
            input=credential_input,
            text=True,
            check=True,
            capture_output=True,
            cwd=DRAIL_REPO_PATH
        )
        
        print("Git credentials configured successfully")
        return True
            
    except subprocess.CalledProcessError as e:
        print(f"Git credential setup failed:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"Error setting up Git credentials: {str(e)}")
        return False

from uuid import getnode

import os
def get_mac_address():
    """Get the MAC address of the first network interface"""
    try:
        # Get all network interfaces' MAC addresses
        mac = ':'.join(['{:02x}'.format((getnode() >> elements) & 0xff) 
                       for elements in range(0,8*6,8)][::-1])
        
        # if sliurm_job_id is present return that 
        if 'SLURM_JOB_ID' in os.environ:
            return os.environ['SLURM_JOB_ID']
        else:
            return mac
    except Exception as e:
        print(f"Error getting MAC address: {e}")
        return None

    
def update_resource_status(status):
    """Update resource status (ready/busy)"""
    try:
        mac_address = get_mac_address()
        if not mac_address:
            return False
            
        response = requests.put(
            web_server_url + '/api/resource/status',
            json={
                'mac_address': mac_address,
                'status': status
            }
        )
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    args = parse_args()
    DRAIL_REPO_PATH = args.repo_path
    MINICONDA_PATH = args.miniconda_path
    ENABLE_GIT_PULL = args.enable_git_pull
    
    print(f"Using Miniconda path: {MINICONDA_PATH}")
    print(f"Using repository path: {DRAIL_REPO_PATH}")
    
    # Set up Git credentials if Git pull is enabled
    if ENABLE_GIT_PULL and not git_setup(args.git_username, args.git_password):
        print("Failed to set up Git credentials. Git pull operations will likely fail.")
    
    while True:
        # Set status to ready when looking for jobs
        update_resource_status('ready')
        print(f"Checking for Jobs at {web_server_url}")

        job = fetch_job()
        if job:
            try:
                job_id = job['id']
                print("Fetched job: ", job)

                # Update job and resource status to running/busy
                update_job_status(job_id, 'running')
                update_resource_status('busy')
                print("Updated job status to running")
                
                # Only perform git pull if enabled
                if ENABLE_GIT_PULL:
                    if not git_pull():
                        print("Git pull failed, proceeding with existing code")
                    else:
                        print("Git pull successful")
                else:
                    print("Git pull disabled")
                
                print(f"Running command: {job['script_name']}")
                success = run_command(job['script_name'])
                
                # Update final status
                if success:
                    update_job_status(job_id, 'completed')
                else:
                    update_job_status(job_id, 'failed')
                
                # Set resource back to ready
                update_resource_status('ready')
                
            except Exception as e:
                print(f"Error processing job {job.get('id', 'unknown')}: {str(e)}")
                if 'id' in job:
                    update_job_status(job['id'], 'failed')
                update_resource_status('ready')
        else:
            print(f"No jobs to run at this time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(1)
        
        time.sleep(5)