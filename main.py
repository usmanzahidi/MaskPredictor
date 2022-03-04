"""
Started by: Usman Zahidi (uz) {16/02/22}

"""

import sys, numpy as np, cv2,argparse,logging
from masks_predictor import MasksPredictor, ClassNames
import matplotlib.pyplot as plt
from os import listdir


parser = argparse.ArgumentParser(description="Test mask predictor")
parser.add_argument("-r", "--rgb", default='', type=str, metavar="PATH", help="path to rgb folder")
parser.add_argument("-d", "--depth", default='', type=str, metavar="PATH", help="path to depth folder")

def call_predictor():

    #args = parser.parse_args()

    #if not args.rgb or not args.depth:
    #    print ('wrong set of arguments');
    #    return

    #rgb_file    = args.rgb
    #depth_file  = args.depth

    model_file  = './model/fp_ss_model.pth'
    config_file = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
    image_dir   ='./images/rgb/'
    depth_dir = './images/depth/'
    rgb_files=listdir(image_dir)
    depth_files=listdir(depth_dir)
    iter=1


    for rgb_file,depth_file in zip(rgb_files,depth_files):

        rgb_image   = cv2.imread(image_dir+rgb_file)
        depth_image = cv2.imread(depth_dir+depth_file)

        if rgb_image is None or depth_image is None:
            message = 'path to rgb or depth image is invalid'
            logging.error(message)
            sys.exit(1)

        if rgb_image.shape != depth_image.shape:
            message = 'rgb and depth image size mismatch'
            logging.error(message)
            sys.exit(1)

        rgbd_image  = np.dstack((rgb_image,depth_image[:,:,0]))

        # list of classes for which depth masks are required, shouldn't be null, returns in the order supplied
        class_list  = [ClassNames.STRAWBERRY,ClassNames.CANOPY,ClassNames.RIGID_STRUCT,ClassNames.BACKGROUND]

        #instantiation
        mask_pred   = MasksPredictor(model_file,config_file)

        # ** main call **
        depth_masks = mask_pred.get_predictions(rgbd_image,class_list)
            #uncomment save_mask_images if wish to write segmented image
            #save_mask_images(rgb_image,depth_masks,image_dir,rgb_file,iter)

        # next process depth_masks for creating otcomaps

        # display for test only
        display_masks(rgbd_image,depth_masks)
        iter += 1

def display_masks(rgbd_image,depth_masks):
    # assuming depth_masks has all 4 masks for display,
    # only for testing
    rgb_image=rgbd_image[:,:,0:3]
    depth_image = rgbd_image[:, :, 3]
    font_sz = 11
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.tick_params(axis='both', which='major', labelsize=font_sz)
    ax1.set_title("Original Image", fontsize=font_sz)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    ax1.imshow(rgb_image)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.tick_params(axis='both', which='major', labelsize=font_sz)
    ax2.set_title("Depth Image", fontsize=font_sz)
    ax2.imshow(depth_image)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.tick_params(axis='both', which='major', labelsize=font_sz)
    ax3.set_title("Strawberry", fontsize=font_sz)
    ax3.imshow(depth_masks[:,:,0])
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.tick_params(axis='both', which='major', labelsize=font_sz)
    ax4.set_title("Canopy", fontsize=font_sz)
    ax4.imshow(depth_masks[:,:,1])
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.tick_params(axis='both', which='major', labelsize=font_sz)
    ax5.set_title("Rigid", fontsize=font_sz)
    ax5.imshow(depth_masks[:,:,2])
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.tick_params(axis='both', which='major', labelsize=font_sz)
    ax6.set_title("Background", fontsize=font_sz)
    ax6.imshow(depth_masks[:,:,3])
    plt.show()

def save_mask_images(rgb_image,depth_masks,image_dir,file_name,iter):

    yellow= depth_masks[:, :, 0].copy() * 255
    blue  = depth_masks[:, :, 3].copy() * 255
    green = depth_masks[:, :, 1].copy() * 255
    red   = depth_masks[:, :, 2].copy() * 255

    bgr_image = depth_masks[:, :, 0:3]
    bgr_image[:, :, 0] = blue
    bgr_image[:, :, 1] = green + yellow
    bgr_image[:, :, 2] = red+ yellow
    bgr_image=np.hstack([rgb_image, bgr_image])
    #save image
    prefix="%04d" % (iter,)
    cv2.imwrite(image_dir+'/segmented_output/'+prefix+'_image.png', bgr_image)

if __name__ == '__main__':
    #example call
    #python main.py -r ./images/rgb/30.png -d ./images/depth/30.png
    call_predictor()


