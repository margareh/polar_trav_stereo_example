# -*- coding: utf-8 -*-
"""
@date: 12-14-2021
@author: Margaret Hansen
@purpose: Define a class for OpenCV stereo processing of images
    with a provided yaml file of parameters
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

class Stereo:
    
    # Constructor
    # [in] file: parameter file (yaml)
    def __init__(self, samples, file=None):
        self.downsample_factor=None
        self.samples = samples
        if file == None:
            file = "../../cfg/stereo_params.yaml"
            print("No param file provided, using default at cfg/stereo_params.yaml")
        self.parseParams(file)

    # Parse stereo param arguments
    # [in] file: parameter file (yaml)
    def parseParams(self, file):
        
        # Load yaml file into a dict
        with open(file) as param_data:
            self.params = yaml.load(param_data, Loader=yaml.FullLoader)

        # Transfer params from dict to variables
        params_incl = list(self.params.keys())

        # Find SGBM param and store as separate variable
        self.sgbm = self.params['SGBM']

        # Necessary params for both methods
        numDisp = (self.params['numDisparities'] if 'numDisparities' in params_incl else 96)
        blSize = (self.params['blockSize'] if 'blockSize' in params_incl else 25)

        # Create stereo object and add individual params
        if self.sgbm:
            self.stereo = cv2.StereoSGBM_create(numDisparities=numDisp, blockSize=blSize)
            
            P1 = (self.params['P1'] if 'P1' in params_incl else 100)
            P2 = (self.params['P2'] if 'P2' in params_incl else 100)
            mode = (self.params['mode'] if 'mode' in params_incl else 0)
            disp12 = (self.params['disp12MaxDiff'] if 'disp12MaxDiff' in params_incl else 1)

            self.stereo.setP1(P1)
            self.stereo.setP2(P2)
            self.stereo.setMode(mode)
            self.stereo.setDisp12MaxDiff(disp12)
        else:
            self.stereo = cv2.StereoBM_create(numDisparities=numDisp, blockSize=blSize)
            
            pfType = (self.params['prefilterType'] if 'prefilterType' in params_incl else 0)
            pfSize = (self.params['prefilterSize'] if 'prefilterSize' in params_incl else 15)
            textThresh = (self.params['textureThreshold'] if 'textureThreshold' in params_incl else 500)

            self.stereo.setPreFilterType(pfType)
            self.stereo.setPreFilterSize(pfSize)
            self.stereo.setTextureThreshold(textThresh)

        # Set remaining optional params that both can use
        minDisp = (self.params['minDisparity'] if 'minDisparity' in params_incl else 0)
        pfCap = (self.params['prefilterCap'] if 'prefilterCap' in params_incl else 30)
        uniqueRatio = (self.params['uniquenessRatio'] if 'uniquenessRatio' in params_incl else 5)
        spWindSize = (self.params['speckleWindowSize'] if 'speckleWindowSize' in params_incl else 5)
        spRange = (self.params['speckleRange'] if 'speckleRange' in params_incl else 10)

        self.stereo.setMinDisparity(minDisp)
        self.stereo.setPreFilterCap(pfCap)
        self.stereo.setUniquenessRatio(uniqueRatio)
        self.stereo.setSpeckleWindowSize(spWindSize)
        self.stereo.setSpeckleRange(spRange)

    # Perform rectification and stereo matching
    # Compute and store disparity and point cloud for a 
    # given pair of images using the provided rectification maps
    # [in] left_img: left image of stereo pair
    # [in] right_img: right image of stereo pair
    # [in] rect_maps: rectification maps for both images
    # [in] plot_disp: whether to plot the resulting disparity
    # [in] plot_rect: whether to plot rectified stereo pair
    # [in] downsample_factor: downsampling factor to use to resize rectified images, defaults to None for same size
    def rectAndStereo(self, left_img, right_img, rect_maps,
                      plot_disp=False, plot_rect=False):
        
        # Rectify images
        left_img_rect = cv2.remap(left_img, rect_maps[0], rect_maps[1], cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT)
        right_img_rect = cv2.remap(right_img, rect_maps[2], rect_maps[3], cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT)
        
        # Print rectified images
        if plot_rect:
            print('Displaying rectified stereo pair')
            fig, (ax1 ,ax2) = plt.subplots(1, 2)
            ax1.imshow(left_img_rect, cmap='gray')
            ax1.axis('off')
            ax2.imshow(right_img_rect, cmap='gray')
            ax2.axis('off')
            plt.show()
        
        # Stereo on rectified image
        disp = (self.stereo.compute(left_img_rect, right_img_rect) / 16.0).astype(np.float32)
        self.disp = disp
        
        # Plot disparity image if asked for
        if plot_disp:
            print('Displaying disparity image from stereo algorithm...')
            plt.imshow(self.disp)
            plt.colorbar()
            plt.show()
    
    # Generate a point cloud from disparity image
    # [in] projMx: projection matrix used to reproject disparity
    # [in] plot_pcl: whether to plot the resulting point cloud
    def getPCL(self, projMx, plot_pcl=False):
        
        # Reproject disparity into 3D
        self.pcl = cv2.reprojectImageTo3D(self.disp, projMx)

        # Reshape to unstructured point cloud
        self.pcl = self.pcl[np.all(~np.isinf(self.pcl),2),:]
        
        if plot_pcl:
            sample = np.random.choice(np.arange(self.pcl.shape[0]), self.samples)
            pcl_sample = self.pcl[sample,:]
            
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pcl_sample[:,0], pcl_sample[:,1], pcl_sample[:,2], s=0.5, c=-pcl_sample[:,2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(-10,10)
            ax.set_ylim(-10,10)
            ax.set_zlim(-10,10)
            # ax.view_init(20, -100)
            plt.show()
    
    # Transform point cloud given translation t and rotation matrix R
    # [in] t: translation vector to apply (added after rotation)
    # [in] R: rotation matrix to apply
    # [in] plot_pcl: whether to plot the resulting point cloud
    def tfPCL(self, t, R, plot_pcl=False):
        self.pcl[np.isinf(self.pcl)] = 0
        # w,h,_ = self.pcl.shape
        # pcl_flat = self.pcl.reshape((w*h,3))
        pcl_rot = (np.matmul(R,self.pcl.T)+t).T
        self.pcl = pcl_rot
        
        if plot_pcl:
            print('Displaying point cloud in world coordinates...')
            sample_rot = np.random.choice(np.arange(self.pcl.shape[0]), self.samples)
            pcl_sample_rot = self.pcl[sample_rot,:]
            
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pcl_sample_rot[:,0], pcl_sample_rot[:,1], pcl_sample_rot[:,2], s=0.5, c=-pcl_sample_rot[:,2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim(-10,10)
            ax.set_ylim(-10,10)
            ax.set_zlim(-10,10)
            # ax.view_init(20, -100)
            plt.show()

# EOF