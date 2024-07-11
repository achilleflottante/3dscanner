import open3d as o3d
import numpy as np


cameramatrice = np.array([[3.44484369e+03, 0.00000000e+00, 2.05789265e+03],
 [0.00000000e+00, 3.18929861e+03, 1.18328600e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
color_raw = o3d.io.read_image(r"D:\\Ecole\\Cours\\info\\scanner\data\\1.jpg")
depth_raw = o3d.io.read_image(r"D:\Ecole\Cours\info\scanner\lighter1.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        2296, 4080, 3.18929861e+03, 3.44484369e+03,  2296/2, 4080/2))

#pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
 #                                                    o3d.camera.PinholeCameraIntrinsic( o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
   
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])