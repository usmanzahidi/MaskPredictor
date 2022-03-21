"""
Started by: Usman Zahidi (uz) {16/02/22}

"""
#general imports
import os, numpy as np, cv2,pickle, logging
from enum                        import Enum,unique
from scipy                       import ndimage

# detectron imports
from detectron2.config           import get_cfg
from detectron2.engine.defaults  import DefaultPredictor
from detectron2                  import model_zoo

# project imports
from visualizer.fastpick_visualizer   import FastPickVisualizer,ColorMode

#temp imports
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MasksPredictor:

    def __init__(self, model_file,config_file,metadata_file,num_classes, scale=1.0, instance_mode=ColorMode.SEGMENTATION):

        self.instance_mode=instance_mode
        self.scale=scale
        self.metadata=self.get_metadata(metadata_file)
        cfg = self.init_config(model_file, config_file, num_classes)

        try:
            self.predictor=DefaultPredictor(cfg)
        except Exception as e:
            logging.error(e)
            raise Exception(e)

    def init_config(self, model_file, config_file, num_classes):
        cfg = get_cfg()
        try:
            cfg.merge_from_file(model_zoo.get_config_file(config_file))
        except Exception as e:
            logging.error(e)
            raise Exception(e)

        cfg.MODEL.WEIGHTS = os.path.join(model_file)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # 1.strawberry, 2. Canopy, 3. Rigid Structure
        return cfg

    def get_metadata(self,metadata_file):

        #metadata file has name of classes, it is created to avoid having custom dataset and taking definitions
        # from annotations, instead. It has structure of MetaDataCatlog output of detectron2
        try:
            file = open(metadata_file, 'rb')
        except Exception as e:
            logging.error(e)
            raise Exception(e)

        data = pickle.load(file)
        file.close()
        return data

    def get_predictions(self,rgbd_image,class_list,output_type):

        if (bool(class_list)==False):
            e='class list empty'
            logging.error(e)
            raise Exception(e)


        depth_image = rgbd_image[:, :, 3]
        rgb_image=rgbd_image[:, :, :3]

        outputs = self.predictor(rgb_image)
        # [16/02/22]:format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = FastPickVisualizer(rgb_image,
                       metadata=self.metadata,
                       scale=self.scale,
                       instance_mode=self.instance_mode
                       )
        predictions = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        pred_masks = predictions.get_image()[:, :, ::-1]
        return self.get_masks(pred_masks, depth_image, class_list,output_type)

    def get_background(self,depth_image):

        depth_zero = (depth_image == 0)*1
        depth_zero = depth_zero * np.amax(depth_image)
        depth_image = depth_image + depth_zero
        bg_mask = depth_image > 5
        return bg_mask

    def get_masks(self,fg_masks, depth_image, class_list,output_type):

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
        plt.imshow(mask)
        blue = np.bitwise_not(mask)

        if output_type==OutputType.COLOR_MASKS:
            depth_image=255

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

@unique
class OutputType(Enum):
    """
    Enum of different class names
    """

    DEPTH_MASKS = 1
    """
    Desired output is depth mask
    """
    COLOR_MASKS = 2
    """
    Desired output is color mask for writing to masks rgb (for displaying etc)
    """