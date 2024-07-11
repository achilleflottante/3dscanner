import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main(image1_path, image2_path):
    # Charger les images
    img1 = cv2.imread('D:\\Ecole\\Cours\\info\\scanner\\Stereo reconstruction\\Test 0_ openCV hsv\\media\\trousse11.jpg')
    img2 = cv2.imread('D:\\Ecole\\Cours\\info\\scanner\\Stereo reconstruction\\Test 0_ openCV hsv\\media\\trousse12.jpg')

    # Convertir en niveaux de gris
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Détection des points d'intérêt et des descripteurs
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    

    # Appariement des points d'intérêt
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Filtre des bonnes correspondances
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Estimation de la matrice fondamentale
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)

    # Rectification
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts, dst_pts, F, (w1, h1))

    img1_rect = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rect = cv2.warpPerspective(img2, H2, (w2, h2))

    # Recherche de disparités
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5)
    disparity = stereo.compute(gray1, gray2)

    # Calcul de la carte de profondeur
    focal_length = 0.8 * img1.shape[1]
    baseline = 5  # Distance entre les deux caméras en centimètres
    Q = np.float32([[1, 0, 0, -0.5 * img1.shape[1]],
                    [0, -1, 0, 0.5 * img1.shape[0]],
                    [0, 0, 0, -focal_length],
                    [0, 0, 1 / baseline, 0]])
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    valid_points_3D = points_3D[~np.any(np.isinf(points_3D), axis=2)]

    return valid_points_3D, disparity

if __name__ == "__main__":
    image1_path = r"C:\Users\louis\cours-info\projet_scannerlowcost\Test chat GPT\gourdeL.jpg"
    image2_path = r"C:\Users\louis\cours-info\projet_scannerlowcost\Test chat GPT\gourdeR.jpg"
    valid_points_3D, disparity = main(image1_path, image2_path)

    print(len(valid_points_3D))
    print(valid_points_3D[0])
    print(valid_points_3D[0][0])
    print(valid_points_3D)

    val = []
    for i in range(len(valid_points_3D)):
        val.append(valid_points_3D[i])
    valid_points_3D = np.array(val)
    #print(valid_points_3D)

    # Affichage de la carte de profondeur
    plt.imshow(disparity, cmap='gray')
    plt.title('Carte de profondeur')
    plt.colorbar()
    plt.show()
    

    # Affichage des points en 3D avec Matplotlib
    nombre_de_points = 100000
    indices = np.linspace(0, len(valid_points_3D)-1, nombre_de_points, dtype=int)
    points_extraits = valid_points_3D[indices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_extraits[:,0], points_extraits[:,1], points_extraits[:,2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()






