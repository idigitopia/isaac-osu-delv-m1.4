# PointConv

This repository contains the code for the PointConv.

## Installation
Install module point_cloud_encoders. This also installs and compiles pointnet2_ops_lib which have the utilities for pointcloud processing.

### Prerequisites
CUDA version 12.4+ is required.
```
    pip install -e .
```

## Usage

`test_pointconv.py` has the PointConv encoder implementation involving multiple PointConv layers.

```
    python point_cloud_encoders/test_pointconv.py
```
