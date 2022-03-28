from enum import Enum,unique

@unique
class ClassNames(Enum):
    """
    Enum of different class names
    """

    STRAWBERRY   = 0,
    """
    Class strawberry, depicted by yellow colour
    """
    CANOPY       = 1,
    """
    Class canopy, depicted by green colour
    """
    RIGID_STRUCT = 2,
    """
    Class rigid structure, depicted by red colour
    """

    BACKGROUND   = 3,
    """
    Class background, depicted by blue colour
    """

    ALL         = 4

@unique
class OutputType(Enum):
    """
    Enum of different class names
    """

    DEPTH_MASKS = 0
    """
    Desired output is depth mask
    """
    COLOR_MASKS = 1
    """
    Desired output is color mask for writing to masks rgb (for presentation images etc)
    """

    POINTCLOUD_DISPLAY = 2
    """
    Desired output is Point Cloud 3D Display
    """