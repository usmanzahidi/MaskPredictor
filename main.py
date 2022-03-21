"""
Started by: Usman Zahidi (uz) {16/02/22}

"""

import sys, numpy as np, cv2,logging
from masks_predictor import MasksPredictor, ClassNames, OutputType
from os import listdir

NUM_CLASSES = 3 #1.strawberry, 2. canopy, 3. rigid

def call_predictor():

    model_file  = './model/fp_ss_model.pth'
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
        #print(image_dir + rgb_file)
        rgb_image   = cv2.imread(image_dir+rgb_file)
        depth_image = cv2.imread(depth_dir+depth_file)

        if rgb_image is None or depth_image is None:
            message = 'path to rgb or depth image is invalid'
            logging.error(message)


        if rgb_image.shape != depth_image.shape:
            message = 'rgb and depth image size mismatch'
            logging.error(message)


        rgbd_image  = np.dstack((rgb_image,depth_image[:,:,0]))

        # list of classes for which depth masks are required, shouldn't be null, returns in the order supplied
        class_list  = [ClassNames.STRAWBERRY,ClassNames.CANOPY,ClassNames.RIGID_STRUCT,ClassNames.BACKGROUND]

        # ** main call **
        try:
            output_masks = mask_pred.get_predictions(rgbd_image,class_list,OutputType.DEPTH_MASKS)
        except Exception as e:
            logging.error(e)
            sys.exit(1)


        # next: process depth_masks for creating otcomaps

        #save_mask_images writes segmented image in output dir
        save_mask_images(rgb_image,output_masks,output_dir,iter)


        iter += 1

def save_mask_images(rgb_image,depth_masks,output_dir,iter):

    yellow= depth_masks[:, :, 0].copy()
    green = depth_masks[:, :, 1].copy()
    red   = depth_masks[:, :, 2].copy()
    blue = depth_masks[:, :, 3].copy()

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


