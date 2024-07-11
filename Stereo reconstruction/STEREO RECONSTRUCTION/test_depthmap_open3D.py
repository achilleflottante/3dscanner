import numpy as np
from PIL import Image
import open3d as o3d

def load_depth_image(image_path):
    """
    Charge une image de profondeur et la convertit en un tableau numpy.

    Paramètres :
    - image_path : chemin du fichier image de profondeur.
    Sortie :
    - depth_image : tableau numpy de l'image de profondeur.
    """
    image = Image.open(image_path)
    depth_image = np.array(image)
    depth_image = - depth_image[:, :, 0]
    depth_image[0,0] += 1000
    return depth_image

def depth_to_point_cloud(depth_image, fx, fy, cx, cy):
    """
    Convertir une image de profondeur en un nuage de points.
    
    Paramètres :
    - depth_image : tableau numpy de l'image de profondeur.
    - fx, fy : longueurs focales dans les directions x et y.
    - cx, cy : décalages du point principal dans les directions x et y.
    Sortie :
    - point_cloud : objet Open3D PointCloud.
    """
    
    height, width = depth_image.shape
    print(height, width)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image / 1000.0  # suppose que l'unité de profondeur est en millimètres et on la mets en mètres

    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud


def main():
    image_path = r"Stereo reconstruction\TEST notion\media\depthmap1.png"
    fx = 525.0  # distance focale en x
    fy = 525.0  # distance focale en y
    cx = 300 # point principal x
    cy = 400  # point principal y

    depth_image = load_depth_image(image_path)
    point_cloud = depth_to_point_cloud(depth_image, fx, fy, cx, cy)

    o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    main()
