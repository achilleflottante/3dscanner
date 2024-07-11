import cv2
import numpy as np
import sys



path1 = r"D:\Ecole\Cours\info\scanner\Stereo reconstruction\Test 0_ openCV hsv\media\trousse11.jpg"
path2 = r"D:\Ecole\Cours\info\scanner\Stereo reconstruction\Test 0_ openCV hsv\media\trousse12.jpg"
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

cv2.imshow('image_1', img1)
cv2.imshow('image_2', img2)

ratio = 50
lar1 = int(img1.shape[1] * ratio / 100)
haut1 = int(img1.shape[0] * ratio / 100)
dim1 = (lar1, haut1)
img1 = cv2.resize(img1, dim1, interpolation = cv2.INTER_AREA)

ratio = 50
lar2 = int(img2.shape[1] * ratio / 100)
haut2 = int(img2.shape[0] * ratio / 100)
dim2 = (lar2, haut2)
img2 = cv2.resize(img2, dim2, interpolation = cv2.INTER_AREA)

hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)





lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask11 = cv2.inRange(hsv1, lower_red1, upper_red1)
mask12 = cv2.inRange(hsv1, lower_red2, upper_red2)
mask1 = cv2.bitwise_or(mask11, mask12)
mask21 = cv2.inRange(hsv2, lower_red1, upper_red1)
mask22 = cv2.inRange(hsv2, lower_red2, upper_red2)
mask2 = cv2.bitwise_or(mask21, mask22)


# Trouver les contours sur le mask
contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for contour in contours1:
    area = cv2.contourArea(contour)
    if area > 1000:
        cv2.drawContours(hsv1, contours1, -1, (0, 255, 0), 1)
        cv2.drawContours(mask1, contours1, -1, (0, 255, 0), 1)

contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for contour in contours2:
    area = cv2.contourArea(contour)
    if area > 1000:
        cv2.drawContours(hsv2, contours2, -1, (0, 255, 0), 1)
        cv2.drawContours(mask2, contours2, -1, (0, 255, 0), 1)



#cv2.imshow('image1', img1)
#cv2.imshow('image2', img2)
cv2.imshow('image_hsv1', hsv1)
cv2.imshow('image_hsv2', hsv2)
cv2.imshow('mask_image1', mask1)
cv2.imshow('mask_image2', mask2)



key = cv2.waitKey(0) & 0x0FF
if key == 27:
    print('arrêt du programme par l\'utilisateur')
    cv2.destroyWindow('image originale')
    sys.exit(0)


