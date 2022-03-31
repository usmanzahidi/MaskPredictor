"""
Started by: Usman Zahidi (uz) {16/02/22}

"""
#general imports
import os, sys, numpy as np, pickle, logging

# detectron imports
from detectron2.config           import get_cfg
from detectron2.engine.defaults  import DefaultPredictor
from detectron2                  import model_zoo

# project imports
try:
    from visualizer.fastpick_visualizer   import FastPickVisualizer,ColorMode
    from visualizer.mask_predictor_enums import ClassNames,OutputType
    from visualizer.pointcloud_visualizer import PointCloudVisualizer
except:
    from .visualizer.fastpick_visualizer   import FastPickVisualizer,ColorMode
    from .visualizer.mask_predictor_enums import ClassNames,OutputType
    from .visualizer.pointcloud_visualizer import PointCloudVisualizer




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
            print(e)
            raise Exception(e)

    def init_config(self, model_file, config_file, num_classes):
        cfg = get_cfg()
        try:
            cfg.merge_from_file(model_zoo.get_config_file(config_file))
        except Exception as e:
            logging.error(e)
            print(e)
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
            print(e)
            raise Exception(e)

        data = pickle.load(file)
        file.close()
        return data

    def get_predictions(self,rgbd_image,class_list,output_type,display_class):

        if (bool(class_list)==False):
            e='class list empty'
            logging.error(e)
            print(e)
            raise Exception(e)


        depth_image = rgbd_image[:, :, 3]
        rgb_image=rgbd_image[:, :, :3].astype(np.uint8)


        outputs = self.predictor(rgb_image)
        # [16/02/22]:format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = FastPickVisualizer(rgb_image,
                       metadata=self.metadata,
                       scale=self.scale,
                       instance_mode=self.instance_mode
                       )
        predictions = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        pred_masks = predictions.get_image()[:, :, ::-1]


        depth_masks,rgb_masks = self.get_masks(pred_masks, rgb_image,depth_image, class_list,output_type)

        if output_type==OutputType.POINTCLOUD_DISPLAY:
            pc_visualizer = PointCloudVisualizer(rgb_masks, depth_masks)
            pc_visualizer.visualize_pcl_mask(display_class,depth_image,self.np_to_py_list(rgb_image))
            sys.exit(0)
        else:
            return depth_masks,rgb_masks

    def np_to_py_list( self, np_arrays ):
        # receives a list of numpy arrays and converts each ndarray to a python (nested) list
        return [ a.tolist() for a in np_arrays ]
        
    def get_background(self,depth_image):

        depth_zero = (depth_image == 0)*1
        depth_zero = depth_zero * np.amax(depth_image)
        depth_image = depth_image + depth_zero
        bg_mask = depth_image > 5
        return bg_mask

    def get_masks(self,fg_masks, rgb_image,depth_image, class_list,output_type):

        # input three foreground class' masks and calculate leftover as background mask
        # then output requested depth masks as per class_list order

        rgb_masks = list()
        h,w,b=rgb_image.shape
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

        if output_type==OutputType.COLOR_MASKS:
            # make depth_image unit scalar
            depth_image=1

        depth_masks=list()

        for classes in class_list:
            if   classes==ClassNames.STRAWBERRY:
                depth_masks.append(yellow*depth_image)
                a=(1 * np.dstack([yellow, yellow, yellow]) * rgb_image)
                rgb_masks.append(a)
            elif classes == ClassNames.CANOPY:
                depth_masks.append(green*depth_image)
                rgb_masks.append((1*np.dstack([green, green, green])*rgb_image))
            elif classes == ClassNames.RIGID_STRUCT:
                depth_masks.append(red*depth_image)
                rgb_masks.append((1*np.dstack([red, red, red])*rgb_image))
            elif classes == ClassNames.BACKGROUND:
                depth_masks.append(blue*depth_image)
                rgb_masks.append((1*np.dstack([blue, blue, blue])*rgb_image))
            elif classes == ClassNames.ALL:
                depth_masks.append(depth_image)
                rgb_masks.append((rgb_image))
        return np.dstack(depth_masks),rgb_masks