# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import open3d as o3d
import cv2
from .mask_predictor_enums import ClassNames


class PointCloudVisualizer():

    def __init__(self,rgb_masks,depth_masks):
        self.depth_masks=depth_masks
        self.rgb_masks = rgb_masks

    def visualize_pcl_mask(self,class_name,depth_image,rgb_image):

        if class_name   == ClassNames.STRAWBERRY:
            rgb_mask    =  self.rgb_masks[class_name.value]
            depth_map   =  self.depth_masks[:, :, class_name.value]

        elif class_name == ClassNames.CANOPY:
            rgb_mask    =  self.rgb_masks[class_name.value]
            depth_map   =  self.depth_masks[:, :, class_name.value]

        elif class_name == ClassNames.RIGID_STRUCT:
            rgb_mask    =  self.rgb_masks[class_name.value]
            depth_map   =  self.depth_masks[:, :, class_name.value]

        elif class_name == ClassNames.BACKGROUND:
            rgb_mask    =  self.rgb_masks[class_name.value]
            depth_map   =  self.depth_masks[:, :, class_name.value]

        elif class_name == ClassNames.ALL:
            rgb_mask    =  rgb_image
            depth_map   =  depth_image

        conv_depth_map = depth_map.copy()
        if class_name != ClassNames.ALL:
            conv_rgb_image = np.asarray(rgb_mask, dtype=np.uint8).copy()
        else:
            conv_rgb_image = rgb_mask.copy()
        h, w, b = conv_rgb_image.shape
        conv_rgb_image = cv2.cvtColor(conv_rgb_image, cv2.COLOR_BGR2RGB)

        conv_rgb_image = np.ascontiguousarray(conv_rgb_image)
        conv_depth_map = np.ascontiguousarray(conv_depth_map.astype(np.int16)) #depth map is 16 bit

        #create open3D images from opencv
        conv_rgb_image = o3d.geometry.Image(conv_rgb_image)
        conv_depth_map = o3d.geometry.Image(conv_depth_map)
        print(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            conv_rgb_image, conv_depth_map, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # uz: Flip it, pointcloud otherwise upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #uz: hardcoded values should be taken from param once supplied (for our test cameras).
        window_title='Fastpick 3D Viewer : Class [' + class_name.name + ']'
        o3d.visualization.draw(pcd,title= window_title,
                               field_of_view=60.0,
                               bg_color=(0.2, 0.2, 0.2, 1.0),
                               lookat=[2.6172, 2.0475, 1.532],
                               up=[-0.0694, -0.9768, 0.2024],
                               width=w,height=h,show_skybox=False,show_ui=True)