"""
Started by: Usman Zahidi (uz) {16/02/22}

"""

import sys, numpy as np, cv2,logging
from masks_predictor import MasksPredictor, ClassNames, OutputType
from os import listdir

NUM_CLASSES = 3 #1.strawberry, 2. canopy, 3. rigid (model finds three classes, leftover is then labelled as background)

def call_predictor():

    model_file  = './model/fp_model.pth'
    config_file = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
    metadata_file ='./data/metadata.pkl'
    image_dir   ='./images/rgb/'
    depth_dir = './images/depth/'
    output_dir='./images/output/'
    rgb_files=listdir(image_dir)
    depth_files=listdir(depth_dir)
    iter=1

    # instantiation
    mask_pred = MasksPredictor(model_file, config_file, metadata_file, NUM_CLASSES)

    #loop for generating/saving segmentation output images
    for rgb_file,depth_file in zip(rgb_files,depth_files):
        rgb_image   = cv2.imread(image_dir+rgb_file)
        depth_image = cv2.imread(depth_dir + depth_file, cv2.IMREAD_UNCHANGED)  #cv2.IMREAD_UNCHANGED to keep uint16 type

        if rgb_image is None or depth_image is None:
            message = 'path to rgb or depth image is invalid'
            logging.error(message)

        # check for h,w match of rgb and depth
        if rgb_image.shape[:2] != depth_image.shape:
            message = 'rgb and depth image size mismatch'
            logging.error(message)


        rgbd_image  = np.dstack((rgb_image.astype(np.uint16),depth_image))

        # list of classes for which depth masks are required, shouldn't be null, returns in the order supplied
        depth_class_list  = [ClassNames.STRAWBERRY,ClassNames.CANOPY,ClassNames.RIGID_STRUCT,ClassNames.BACKGROUND]

        #display_class works only with OutputType.POINTCLOUD_DISPLAY type
        display_class = ClassNames.ALL  # ClassNames.ALL shows all classes in point cloud, change class as required

        # ** main call **
        try:
            depth_masks, rgb_masks = mask_pred.get_predictions(rgbd_image,depth_class_list,OutputType.DEPTH_MASKS,
                                                               display_class)

        except Exception as e:
            logging.error(e)
            print(e)
            sys.exit(1)


        # next: process depth_masks for creating otcomaps

        #save_mask_images writes segmented image in output dir
        #save_mask_images(rgb_image,depth_masks,output_dir,iter)


        iter += 1

#normalize data, required for saving mask images
def nor(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def save_mask_images(rgb_image,depth_masks,output_dir,iter):

    yellow= nor(depth_masks[:, :, 0].copy())*255
    green = nor(depth_masks[:, :, 1].copy())*255
    red   = nor(depth_masks[:, :, 2].copy())*255
    blue  = nor(depth_masks[:, :, 3].copy())*255

    bgr_image = depth_masks[:, :, 0:3].copy()
    bgr_image[:, :, 0] = blue
    bgr_image[:, :, 1] = green + yellow
    bgr_image[:, :, 2] = red + yellow
    bgr_image=np.hstack([rgb_image, bgr_image])
    prefix="%04d" % (iter,)
    output_filename=output_dir + prefix + '_image.png'
    print(output_filename)
    cv2.imwrite(output_filename, bgr_image)

if __name__ == '__main__':
    call_predictor()


