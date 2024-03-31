import numpy as np

file_path = 'calibration/stereo_calibration.npz'
# file_path = 'calibration/realsense_calibration.npz'

with np.load(file_path) as calibration:
    camera_matrix_left = calibration['camera_matrix_left']
    dist_coeffs_left = calibration['dist_coeffs_left']
    camera_matrix_right = calibration['camera_matrix_right']
    dist_coeffs_right = calibration['dist_coeffs_right']
    R = calibration['R']
    T = calibration['T']

print("\nLeft Camera Matrix:\n", camera_matrix_left)
print("\nLeft Distortion Coefficients:\n", dist_coeffs_left)
print("\nRight Camera Matrix:\n", camera_matrix_right)
print("\nRight Distortion Coefficients:\n", dist_coeffs_right)
print("\nRotation Matrix:\n", R)
print("\nTranslation Vector:\n", T)