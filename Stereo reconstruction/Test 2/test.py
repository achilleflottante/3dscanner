import cv2
import numpy as np
import matplotlib.pyplot as plt




## 1

img1 = cv2.imread('D:\\Ecole\\Cours\\info\\scanner\\Stereo reconstruction\\Test 2\\dataset\\maxL.jpg')
img2 = cv2.imread('D:\\Ecole\\Cours\\info\\scanner\\Stereo reconstruction\\Test 2\\dataset\\maxR.jpg')

fx, fy = 3, 3
cx, cy = 1920/2, 1080/2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Camera intrinsic matrix



## 2

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)



## 3

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)



## 4

# Build the projection matrices for the two cameras
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = np.hstack((R, t))

# Convert the projection matrices to the camera coordinate system
P1 = K @ P1
P2 = K @ P2

# Triangulate the 3D points
points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
points_3D = points_4D / points_4D[3]  # Convert from homogeneous to Cartesian coordinates
points_3D = points_3D[:3, :].T



## 5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D points
ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], marker='o', s=5, c='r', alpha=0.5)

# Configure the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()





