# Stereo Example for POLAR Traverse Dataset

This repository contains a simple example of OpenCV's SGBM stereo algorithm run on the [POLAR Traverse Dataset](https://ti.arc.nasa.gov/dataset/PolarTrav/).

The images used for this simple example are from the following subset of the dataset:
- View 1 (forward-facing, light at 20 degrees)
- Traverse 3 (cameras at 1.35 m height with 35 degree pitch)

5 total image pairs are used with the following additional parameters:
- 1 m along test bed, 25 ms exposure
- 9 m along test bed, 5 ms exposure
- 9 m along test bed, 25 ms exposure
- 9 m along test bed, 75 ms exposure
- 9 m along test bed, 300 ms exposure

## Structure

The `gen_disparity.py` script in the `scripts` subdirectory can be used to produce the results. This script calls the stereo matcher class defined in `stereo.py` to perform image rectification and matching using OpenCV, followed by computation of disparities and generation of point clouds.

The input data is contained in the `data` subdirectory. Results will appear in the `results` subdirectory.

The stereo algorithm parameters and the camera calibration information are found in the `cfg` and `calib` subdirectories, respectively.

## Depends

This code was run using Python 3.8.10 on Ubuntu 20.04 with the following additional packages:

- OpenCV (stereo) >= 4.2.0
- Open3D (point cloud processing) >= 0.17.0
- numpy >= 1.24.3
- matplotlib >= 3.7.1
- yaml >= 5.3.1

