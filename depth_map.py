import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob


# Test Variables
test_path = 'test_images/'
cal_path = 'calibration/realsense_calibration.npz'
dist = '2000mm - IR'

# Testing
mouse_pt = (0, 0)

def mouse_loc(event, x, y, args, params):
    global mouse_pt
    mouse_pt = (x, y)

cv.namedWindow('Preview')
cv.setMouseCallback('Preview', mouse_loc)

class depthMap:
    def __init__(self, calibration, image_size):
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi1, self.roi2 = cv.stereoRectify(
            calibration['camera_matrix_left'],
            calibration['dist_coeffs_left'],
            calibration['camera_matrix_right'],
            calibration['dist_coeffs_right'],
            image_size,
            calibration['R'],
            calibration['T']
        )

        self.left_map1, self.left_map2 = cv.initUndistortRectifyMap(
            calibration['camera_matrix_left'],
            calibration['dist_coeffs_left'],
            self.R1,
            self.P1,
            image_size,
            cv.CV_32FC1
        )

        self.right_map1, self.right_map2 = cv.initUndistortRectifyMap(
            calibration['camera_matrix_right'],
            calibration['dist_coeffs_right'],
            self.R2,
            self.P2,
            image_size,
            cv.CV_32FC1
        )

        self.numDisparities = 16*2
        self.blockSize = 35

        self.stereo = cv.StereoBM.create(numDisparities=self.numDisparities, blockSize=self.blockSize)
        self.stereo.setSpeckleRange(16)
        self.stereo.setSpeckleWindowSize(100)
    
    def computeDisparity(self, left, right):
        # Undistort
        undistort_left = cv.remap(left, self.left_map1, self.left_map2, cv.INTER_LINEAR)
        undistort_right = cv.remap(right, self.right_map1, self.right_map2, cv.INTER_LINEAR)

        # Compute disparity
        disparity = self.stereo.compute(undistort_left, undistort_right)
        
        # Rescale disparity
        disparity = disparity.astype(np.float32) / 16.0

        return disparity

    def computeDepth(self, left, right):
        # Compute disparity
        disparity = self.computeDisparity(left, right)

        # Compute 3D Point Map
        pointMap = cv.reprojectImageTo3D(disparity, self.Q, handleMissingValues=True)

        # Convert units to mm
        pointMap[:, :, 2] = pointMap[:, :, 2] * 1000

        return pointMap

    def getUsableROI(self):
        return cv.getValidDisparityROI(self.roi1, self.roi2, 0, self.numDisparities, self.blockSize)
    

if __name__ == '__main__':
    # Load Calibration File
    calibration = np.load(cal_path)

    # Create Depth Map Object
    dm = depthMap(calibration, (1280, 720))

    # Load Test Images
    color_images = glob.glob('{}{}/*_color.jpg'.format(test_path, dist))
    depth_images = glob.glob('{}{}/*_depth.npy'.format(test_path, dist))
    left_images = glob.glob('{}{}/*_left.jpg'.format(test_path, dist))
    right_images = glob.glob('{}{}/*_right.jpg'.format(test_path, dist))

    for color_img, depth_img, left_img, right_img in zip(color_images, depth_images, left_images, right_images):
        color = cv.imread(color_img)
        depth = np.load(depth_img)
        left = cv.imread(left_img)
        right = cv.imread(right_img)

        left = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
        right = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

        # Compute depth map
        disparity = dm.computeDisparity(left, right)
        est_depth = dm.computeDepth(left, right)

        while True:
            preview = left.copy()
            preview = cv.applyColorMap(preview, colormap=cv.COLORMAP_CIVIDIS)

            ref_dist = depth[mouse_pt[1], mouse_pt[0]]
            est_dist = np.int32(est_depth[mouse_pt[1], mouse_pt[0], 2])

            ref_depth_norm = cv.applyColorMap(np.uint8(cv.normalize(depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)), colormap=cv.COLORMAP_CIVIDIS)
            disparity_norm = cv.applyColorMap(np.uint8(cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)), colormap=cv.COLORMAP_CIVIDIS)

            roi = dm.getUsableROI()
            pt1 = (roi[0], roi[1])
            pt2 = (roi[2], roi[3])
            cv.rectangle(preview, pt1, pt2, (0, 0, 255))
            cv.rectangle(ref_depth_norm, pt1, pt2, (0, 0, 255))
            cv.rectangle(disparity_norm, pt1, pt2, (0, 0, 255))
            cv.circle(preview, mouse_pt, 4, (0, 0, 255))
            cv.circle(ref_depth_norm, mouse_pt, 4, (0, 0, 255))
            cv.circle(disparity_norm, mouse_pt, 4, (0, 0, 255))
            cv.putText(preview, 'Ref: {}mm'.format(ref_dist), (mouse_pt[0], mouse_pt[1]), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            cv.putText(preview, 'Est: {}mm'.format(est_dist), (mouse_pt[0], mouse_pt[1] + 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            # Display Color
            cv.imshow('Preview', preview)
            cv.imshow('Reference Depth', ref_depth_norm)
            cv.imshow('Estimation Disparity', disparity_norm)
        
            # q to quit
            if cv.waitKey(1) == ord('q'):
                break

        # Destroy all windows
        cv.destroyAllWindows()
