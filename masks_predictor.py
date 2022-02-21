"""
Started by: Usman Zahidi (uz) {16/02/22}

"""

#general imports
import os, sys, numpy as np, cv2,pickle, logging
from enum                        import Enum,unique
from skimage.transform           import resize

# detectron imports
from detectron2.config           import get_cfg
from detectron2.engine.defaults  import DefaultPredictor
from detectron2                  import model_zoo

# project imports
from fastpick_visualizer         import FastPickVisualizer,ColorMode


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class MasksPredictor:

    def __init__(self, model_file,config_file,num_classes=3, scale=1.0, instance_mode=ColorMode.SEGMENTATION):

        self.instance_mode=instance_mode
        self.scale=scale
        self.metadata=self.get_metadata()

        cfg=self.init_config(model_file, config_file, num_classes)

        try:
            self.predictor=DefaultPredictor(cfg)
        except:
            message = f'{sys.exc_info()[0]} occured'
            logging.error(message)
            sys.exit(1)

    def init_config(self, model_file, config_file, num_classes=3):

        cfg = get_cfg()
        try:
            cfg.merge_from_file(model_zoo.get_config_file(config_file))
        except:
            message=f'{sys.exc_info()[0]} occured'
            logging.error(message)
            sys.exit(1)

        cfg.MODEL.WEIGHTS = os.path.join(model_file)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # 1.strawberry,2.canopy, 3.rigid
        return cfg

    def get_metadata(self):

        #metadata file has name of classes, it is created to avoid having custom dataset and taking definitions
        # from annotations, instead. It has structure of MetaDataCatlog output of detectron2
        try:
            file = open('./data/metadata.pkl', 'rb')
        except:
            message = f'{sys.exc_info()[0]} occured'
            logging.error(message)
            sys.exit(1)


        data = pickle.load(file)
        file.close()
        return data

    def get_predictions(self,rgbd_image,class_list):

        # models trained for downsampled images, we first downsample them then predict masks and upsample binary mask

        if (bool(class_list)==False):
            logging.error('class list empty')
            sys.exit(1)

        image_dims=rgbd_image.shape
        up_dims = [int(image_dims[0]), int(image_dims[1])]
        if (up_dims[1] != 1280 or up_dims[0] != 720):
            message='Image resolution not supported by the model, try (1280x720)'
            logging.error(message)
            sys.exit(1)
        down_dims = [int(image_dims[0]/2),int(image_dims[1]/2)]

        depth_image = rgbd_image[:, :, 3]
        rgbd_image=self.downsample_image(rgbd_image,down_dims)
        rgb_image=rgbd_image[:, :, :3]

        outputs = self.predictor(rgb_image)
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = FastPickVisualizer(rgb_image,
                       metadata=self.metadata,
                       scale=self.scale,
                       instance_mode=self.instance_mode
                       )
        predictions = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        pred_masks = predictions.get_image()[:, :, ::-1]
        return self.get_masks(pred_masks, depth_image,class_list, up_dims)


    def get_masks(self,fg_masks, depth_image, class_list, up_dims):

        # input three foreground class' masks and calculate leftover as background mask
        # then output requested depth masks as per class_list order

        # fetch yellow mask (Strawberry)
        yellow = np.bitwise_and(fg_masks[:, :, 1] == 255, fg_masks[:, :, 2] == 255)
        yellow = np.bitwise_and(yellow == True, fg_masks[:, :, 0] == 0)

        #fetch green mask   (Canopy)
        green = np.bitwise_and(fg_masks[:, :, 0] == 0, fg_masks[:, :, 2] == 0)
        green = np.bitwise_and(green == True, fg_masks[:, :, 1] == 255)

        # fetch red mask (Rigid Struct.)
        red = np.bitwise_and(fg_masks[:, :, 0] == 0, fg_masks[:, :, 1] == 0)
        red = np.bitwise_and(red == True, fg_masks[:, :, 2] == 255)

        # create blue (Background)
        mask = np.bitwise_or(red, green)
        mask = np.bitwise_or(mask, yellow)
        blue = np.bitwise_not(mask)


        img = np.zeros(fg_masks.shape)
        img[:, :, 0] = 255
        blue_image = (np.dstack((blue, blue, blue)) * 1)
        img = np.multiply(img, blue_image)

        mask_img          = fg_masks
        mask_img[:, :, 0] = mask
        mask_img[:, :, 1] = mask
        mask_img[:, :, 2] = mask

        mask_img[:, :, 0] = img[:, :, 0]
        mask_img[:, :, 1] = img[:, :, 1]
        mask_img[:, :, 2] = img[:, :, 2]

        #upsample masks
        yellow = self.upsample_masks(yellow, up_dims)
        green  = self.upsample_masks(green, up_dims)
        red    = self.upsample_masks(red, up_dims)
        blue   = self.upsample_masks(blue, up_dims)

        depth_masks=list()
        for classes in class_list:
            if   classes==ClassNames.STRAWBERRY:
                depth_masks.append(yellow*depth_image)
            elif classes == ClassNames.CANOPY:
                depth_masks.append(green*depth_image)
            elif classes == ClassNames.RIGID_STRUCT:
                depth_masks.append(red*depth_image)
            elif classes == ClassNames.BACKGROUND:
                depth_masks.append(blue*depth_image)
        return (np.dstack(depth_masks))


    def downsample_image(self,rgbd_image,dims):
        return cv2.resize(rgbd_image, dsize=(dims[0],dims[1]), interpolation=cv2.INTER_LINEAR)


    def upsample_masks(self,binary_mask,dims):
        return resize(binary_mask, (dims[0], dims[1]), order=0, preserve_range=True)

@unique
class ClassNames(Enum):
    """
    Enum of different class names
    """

    STRAWBERRY   = 1
    """
    Class strawberry, depicted by yellow colour
    """
    CANOPY       = 2
    """
    Class canopy, depicted by green colour
    """
    RIGID_STRUCT = 3
    """
    Class rigid structure, depicted by red colour
    """

    BACKGROUND   = 4
    """
    Class background, depicted by blue colour
    """



