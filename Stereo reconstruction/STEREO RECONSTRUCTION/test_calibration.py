# Chat gpt 19/06/2024

import numpy as np
import cv2
import glob

def calibrate_camera(images_path, pattern_size):
    """
    Calibrate the camera using images of a chessboard pattern.

    Parameters:
    - images_path: Path to the images of the chessboard pattern.
    - pattern_size: Tuple representing the number of inner corners per a chessboard row and column (rows, columns).
    
    Returns:
    - ret: RMS re-projection error.
    - mtx: Camera matrix.
    - dist: Distortion coefficients.
    - rvecs: Rotation vectors.
    - tvecs: Translation vectors.
    """
    # Criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[1], 0:pattern_size[0]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    images = glob.glob(images_path)
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return ret, mtx, dist, rvecs, tvecs

# Path to the calibration images
images_path = 'media/damier1.jpg'

# Number of inner corners per chessboard row and column
pattern_size = (7, 6)  # Adjust this based on your calibration board

ret, mtx, dist, rvecs, tvecs = calibrate_camera(images_path, pattern_size)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
