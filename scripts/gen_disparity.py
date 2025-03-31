# -*- coding: utf-8 -*-
"""
@date: 08-10-2022
@author: Margaret Hansen
@purpose: Test stereo on lunar lab data
"""

# Packages
import argparse
import os
import cv2
import open3d as o3d
import numpy as np

from stereo import Stereo

# Parse arguments
#   [out] args: arguments object
def parse_args():
    
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='../data', help='File path to stereo pair data')
    parser.add_argument('--outpath', type=str, default='../results', help='File path to store results in')
    parser.add_argument('--calib', type=str, default='../calib', help='Path to calibration files to use in stereo processing')
    parser.add_argument('--stereo_params', type=str, default='../cfg/stereo_params.yaml', help='YAML file with config parameters for stereo alg')
    parser.add_argument('--verbose', action='store_true', help='Whether to print verbose messages')
    parser.add_argument('--height', type=float, default=1.35, help='Height of cameras')
    parser.add_argument('--pitch', type=float, default=35, help='Pitch of cameras')
    parser.add_argument('--samples', type=int, default=10000, help='Number of points to sample to display point clouds')
    args = parser.parse_args()
    
    # Add disparity and point cloud output file paths
    args.disp_out_path = args.outpath + '/disp'
    args.pcl_out_path = args.outpath + '/pcl'

    # Create out directory if it doesn't exist
    if os.path.exists(args.outpath) == False:
        os.makedirs(args.outpath)
    
    if os.path.exists(args.disp_out_path) == False:
        os.makedirs(args.disp_out_path)

    if os.path.exists(args.pcl_out_path) == False:
        os.makedirs(args.pcl_out_path)

    # Print args if verbose
    if args.verbose:
        print(args)
        print("\n")

    return args


# Function for displaying inliers and outliers in open3d point cloud
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

# Parse file names into pairs of file names for given directory
#   [in] dir: directory to get stereo pairs from
#   [out] file_list; list of file pairs to use in stereo
def parse_file_names(dir, verbose):

    files = os.listdir(dir)
    left_files = [f for f in files if f.find('cam0') > 0]
    file_list = [[f, f.replace('cam0', 'cam1')] for f in left_files]

    if verbose:
        print("List of stereo pairs:")
        print(file_list)
        print("\n")

    return file_list

def parse_calib_files(calib, dims=(2048,2048)):

    # extrinsics
    ext_fs = cv2.FileStorage(os.path.join(calib, 'extrinsics.yml'), cv2.FILE_STORAGE_READ)
    rot_mat = ext_fs.getNode('rotation_matrix').mat()
    transl_vec = ext_fs.getNode('translation_vector').mat()
    ext_fs.release()

    # left intrinsics
    left_fs = cv2.FileStorage(os.path.join(calib, 'left_intrinsics.yml'), cv2.FILE_STORAGE_READ)
    cam_mx0 = left_fs.getNode('camera_matrix').mat()
    dist0 = left_fs.getNode('distortion_coefficients').mat()
    left_fs.release()

    # right intrinsics
    right_fs = cv2.FileStorage(os.path.join(calib, 'right_intrinsics.yml'), cv2.FILE_STORAGE_READ)
    cam_mx1 = right_fs.getNode('camera_matrix').mat()
    dist1 = right_fs.getNode('distortion_coefficients').mat()
    right_fs.release()

    # Put it all together in the appropriate format
    calib_data = {
        'camera_matrix0': np.array(cam_mx0),
        'camera_matrix1': np.array(cam_mx1),
        'distortion_coeffs0': np.array(dist0),
        'distortion_coeffs1': np.array(dist1),
        'rotation_matrix': np.array(rot_mat),
        'translation_vector': np.array(transl_vec),
        'img_dims': dims
    }

    # # Put it all together in the appropriate format
    # calib_data = {   
    #             'camera_matrix0': np.array([[1.45271e+03, 0.00000000e+00, 0.99953e+03],
    #                                         [0.00000000e+00, 1.45288e+03, 1.03540e+03],
    #                                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    #             'camera_matrix1': np.array([[1.45572e+03, 0.00000000e+00, 1.02112e+03],
    #                                         [0.00000000e+00, 1.45513e+03, 1.01076e+03],
    #                                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    #             'distortion_coeffs0': np.array([[-0.016834, -0.027914, -0.000321, -0.000487, -0.001499]]),
    #             'distortion_coeffs1': np.array([[-0.017925, -0.019475, -0.000444, -0.000287, -0.011515]]),
    #             'rotation_matrix': np.array([[ 9.999957824489283e-01, 1.287498211802540e-04,  2.901466498043608e-03],
    #                                          [-1.384127605741120e-04, 9.999944445926076e-01, -3.330409258625485e-03],
    #                                          [-2.901021589618670e-03, 3.330796812442055e-03,  9.999902448855843e-01]]),
    #             'translation_vector': np.array([[-399.577424 / 1000],
    #                                             [0.167072 / 1000],
    #                                             [-0.584272 / 1000]]),
    #             'img_dims': dims
    #         }

    return calib_data

# Perform stereo matching and return point cloud in camera frame
# Saves point clouds and disparity images to pcd files
#   [in] pairs: List of stereo pair file names to use
#   [in] calib: Calibration information
#   [in] args: Command line arguments
def perform_stereo(pairs, calib, args):

    # Parse args
    params = args.stereo_params
    disp_path = args.disp_out_path
    pcl_path = args.pcl_out_path
    verbose = args.verbose

    # Generate rectification maps
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(calib['camera_matrix0'],
                                                calib['distortion_coeffs0'],
                                                calib['camera_matrix1'],
                                                calib['distortion_coeffs1'],
                                                calib['img_dims'],
                                                calib['rotation_matrix'],
                                                calib['translation_vector'],
                                                alpha=0)
    
    mapL1, mapL2 = cv2.initUndistortRectifyMap(calib['camera_matrix0'],
                                                calib['distortion_coeffs0'],
                                                R1, P1, calib['img_dims'],
                                                cv2.CV_16SC2) # CV_16SC2 is more compact than CV_32FC1
    
    mapR1, mapR2 = cv2.initUndistortRectifyMap(calib['camera_matrix1'],
                                                calib['distortion_coeffs1'],
                                                R2, P2, calib['img_dims'],
                                                cv2.CV_16SC2)
    
    rect_maps = [mapL1, mapL2, mapR1, mapR2]

    # Create stereo object
    if verbose:
        print("Using stereo params:")
        print(params)
    stereo_matcher = Stereo(args.samples, params)

    # Loop through list of pairs and produce point clouds and disparity images
    n = len(pairs)
    for i in range(n):

        # get info for saving images
        img_name = pairs[i][0]
        dist = img_name[img_name.find('_')+1:img_name.find('cam')-2]
        exp = img_name[img_name.find('cam')+5:img_name.find('ms')]
        disp_file = 'disp_'+dist+'m_'+exp+'ms.png'
        pcl_file = 'pcl_'+dist+'m_'+exp+'ms.pcd'

        if verbose:
            print("On stereo pair "+str(i+1)+" out of "+str(n))
            print("   Left image file is "+pairs[i][0])
            print("   Saving disparity to "+disp_file)
            print("   Saving point cloud to "+pcl_file)
            print("\n")

        # Load images
        left_img = cv2.imread(args.datapath+'/'+pairs[i][0], cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(args.datapath+'/'+pairs[i][1], cv2.IMREAD_GRAYSCALE)

        # Get rectified images and disparity
        stereo_matcher.rectAndStereo(left_img, right_img, rect_maps, True, True)
        cv2.imwrite(os.path.join(disp_path, disp_file), stereo_matcher.disp)

        # Get point cloud
        # print(Q)
        stereo_matcher.getPCL(Q)
        # print(stereo_matcher.pcl.shape)
        # print(stereo_matcher.pcl[0:10,:])

        # transform the point cloud into the robot frame (i.e. something I can look at that makes sense)
        t = np.array([[0], [0], [args.height]])
        c = np.cos(-np.deg2rad(args.pitch+90))
        s = np.sin(-np.deg2rad(args.pitch+90))
        R = np.array([[1, 0,  0],
                      [0, c, -s],
                      [0, s,  c]])
        stereo_matcher.tfPCL(t, R)

        # crop the point cloud
        bound = lambda a,b,i: (stereo_matcher.pcl[:,i] >= a)*(stereo_matcher.pcl[:,i] <= b)
        cond = bound(-10,10,0)*bound(0.5,10,1)*bound(-1,1,2)
        stereo_matcher.pcl = stereo_matcher.pcl[cond,:]
        # print(stereo_matcher.pcl.shape)

        # convert to open3d point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stereo_matcher.pcl)

        # downsample
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        # pcd = pcd.uniform_down_sample(every_k_points=5)
        # print(np.asarray(pcd.points).shape)

        # remove outliers
        # pcd_in, inds = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=1.5)
        pcd_in, _ = pcd.remove_radius_outlier(nb_points=5, radius=0.1)
        pcd_in2, _ = pcd_in. remove_statistical_outlier(nb_neighbors=10, std_ratio = 3.0)
        # pcd_in2, inds = pcd_in.remove_statistical_outlier(nb_neighbors=5, std_ratio=2.0)
        # print(np.asarray(pcd.points).shape)
        # display_inlier_outlier(pcd_in, inds)
        o3d.visualization.draw_geometries([pcd_in2])

        # write point cloud to file
        # print(stereo_matcher.pcl.shape)
        o3d.io.write_point_cloud(os.path.join(pcl_path, pcl_file), pcd_in2)


if __name__ == "__main__":
    
    # Parse args and create list of files
    args = parse_args()
    pairs = parse_file_names(args.datapath, args.verbose)

    # Get calibration information
    calib = parse_calib_files(args.calib)
    # print(calib)

    # Perform stereo over set of files
    perform_stereo(pairs, calib, args)


# EOF