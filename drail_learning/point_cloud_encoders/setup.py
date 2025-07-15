# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys

class CustomBuildExt(build_ext):
    def run(self):
        print("\n=== Starting CustomBuildExt... ===")
        print(f"Current working directory: {os.getcwd()}")
        
        # Build the pointnet2_ops_lib
        pointnet2_dir = os.path.join('point_cloud_encoders','pointconv', 'utils', 'pointnet2_ops_lib')
        print(f"\nPointnet2 dir path: {pointnet2_dir}")
        print(f"Directory exists: {os.path.exists(pointnet2_dir)}")
        
        if not os.path.exists(pointnet2_dir):
            raise FileNotFoundError(f"Could not find directory: {pointnet2_dir}")
        
        # Instead of using pip, run python setup.py directly
        current_dir = os.getcwd()
        try:
            print(f"\nChanging to directory: {pointnet2_dir}")
            os.chdir(pointnet2_dir)
            print("Starting pointnet2_ops installation...")
            subprocess.check_call([sys.executable, 'setup.py', 'install'])
            print("Finished pointnet2_ops installation")
        except subprocess.CalledProcessError as e:
            print(f"Error installing pointnet2_ops: {e}")
            raise
        finally:
            print(f"Changing back to: {current_dir}")
            os.chdir(current_dir)
            
        print("\nRunning parent build_ext...")
        build_ext.run(self)
        print("=== CustomBuildExt completed ===\n")

# Dummy extension to trigger build_ext
dummy_ext = Extension(
    name='_dummy',
    sources=['dummy.c'],
    optional=True
)

if __name__ == '__main__':
    print("Starting setup...")
    setup(
        name="point_cloud_encoders",
        packages=find_packages(),
        ext_modules=[dummy_ext],
        cmdclass={
            'build_ext': CustomBuildExt,
        },
    )
    print("Setup completed")
