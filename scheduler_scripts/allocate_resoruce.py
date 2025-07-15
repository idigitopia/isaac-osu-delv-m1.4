import requests
import argparse
import socket
from datetime import datetime
import subprocess
import json
import getpass
from uuid import getnode


endpoint = "drail.ngrok.dev"
def get_available_gpu():
    """Get the first available GPU name using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name,memory.used', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        gpus = result.stdout.strip().split('\n')
        for gpu in gpus:
            name, memory_used = gpu.split(', ')
            if float(memory_used) < 100:  # Consider GPU available if using less than 100MB
                return name
        return None
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        return None

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

def allocate_resource(allocated_by):
    """
    Allocate a resource through the API.
    
    Args:
        allocated_by (str): Username of the person allocating the resource
    
    Returns:
        bool: True if allocation was successful, False otherwise
    """
    name = socket.gethostname()
    gpu_name = get_available_gpu()
    mac_address = get_mac_address()
    
    if gpu_name is None:
        print("No available GPU found")
        return False
        
    if mac_address is None:
        print("Could not get MAC address")
        return False
        
    try:
        response = requests.post(
            f'https://{endpoint}/api/resources/allocate',
            json={
                'name': name,
                'gpu_name': gpu_name,
                'allocated_by': allocated_by,
                'mac_address': mac_address
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Resource allocated successfully! Resource ID: {result['resource_id']}")
            return True
        else:
            print(f"Failed to allocate resource. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error allocating resource: {e}")
        return False

def send_resource_heartbeat():
    """Simple periodic status update to indicate resource is online"""
    try:
        mac_address = get_mac_address()
        if not mac_address:
            return False
            
        response = requests.put(
            f'https://{endpoint}/api/resource/heartbeat',
            json={'mac_address': mac_address}
        )
        return response.status_code == 200
    except:
        return False
    
from time import sleep
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Allocate a GPU resource')
    parser.add_argument('--user', default=getpass.getuser(),
                       help='Username of the person allocating the resource (defaults to current system user)')
    
    args = parser.parse_args()
    
    allocate_resource(args.user)

    print("All allocated resources:")
    response = requests.get(f'https://{endpoint}/api/resources')
    print(response.json())

    while True:
        send_resource_heartbeat()
        sleep(10)
        print(f"Updated resource status: {datetime.now()}")
