import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(image1_path, image2_path):
    # Charger les images
    img1 = cv2.imread('D:\\Ecole\\Cours\\info\\scanner\\Stereo reconstruction\\Test 0_ openCV hsv\\media\\trousse11.jpg')
    img2 = cv2.imread('D:\\Ecole\\Cours\\info\\scanner\\Stereo reconstruction\\Test 0_ openCV hsv\\media\\trousse12.jpg')

    # Convertir en niveaux de gris
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Effectuer la calibration de la caméra et obtenir les paramètres intrinsèques
    # (vous devez remplacer ces étapes par la vraie calibration de la caméra)
    # Intrinsics_left, DistCoeffs_left = calibrate_camera(img1)
    # Intrinsics_right, DistCoeffs_right = calibrate_camera(img2)

    # Exemple de valeurs de paramètres intrinsèques (à remplacer par les vrais résultats de calibration)
    Intrinsics_left = np.array([[800, 0, img1.shape[1] / 2],
                                 [0, 800, img1.shape[0] / 2],
                                 [0, 0, 1]])
    Intrinsics_right = np.array([[800, 0, img2.shape[1] / 2],
                                  [0, 800, img2.shape[0] / 2],
                                  [0, 0, 1]])

    # Convertir en niveaux de gris
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Effectuer la rectification stéréo
    imageSize = gray1.shape[::-1]
    retval, H1, H2 = cv2.stereoRectify(cameraMatrix1=Intrinsics_left,
                                       distCoeffs1=None,
                                       cameraMatrix2=Intrinsics_right,
                                       distCoeffs2=None,
                                       imageSize=imageSize,
                                       R=np.eye(3),
                                       T=np.array([0, 0, 0]),
                                       flags=cv2.CALIB_ZERO_DISPARITY)

    # Rectifier les images
    img1_rect = cv2.warpPerspective(img1, H1, imageSize)
    img2_rect = cv2.warpPerspective(img2, H2, imageSize)

    # Convertir en niveaux de gris
    gray1_rect = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY)
    gray2_rect = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY)

    # Recherche de disparités
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray1_rect, gray2_rect)

    return disparity

if __name__ == "__main__":
    image1_path = r"C:\Users\louis\cours-info\projet_scannerlowcost\Test chat GPT\trousseL.jpg"
    image2_path = r"C:\Users\louis\cours-info\projet_scannerlowcost\Test chat GPT\trousseR.jpg"
    disparity = main(image1_path, image2_path)

    # Affichage de la carte de profondeur
    plt.imshow(disparity, cmap='gray')
    plt.title('Carte de profondeur')
    plt.colorbar()
    plt.show()