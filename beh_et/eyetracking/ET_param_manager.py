import numpy as np
import AnalysisHelpers

""" Eye-Tracking Parameters and Settings

This module includes all the parameters and settings required for eye-tracking analysis of experiment 2.
It includes both lab-specific and lab-agnostic data, and all the methods required to generate a "params" object
which includes individual subject's ET session params.

It contains the following functions:
* get_screen_dims: retrieves the subject's session actual screen dimensions as recorded in the exp.2 log data
* InitParams: initializes the params for a specific session
* UpdateParams: updates parameters for a specific session based on additional calculated data
* GetStimCoords: calculates the coordinates of the stimuli locations with respect to given specifications

@authors: RonyHirsch, AbdoSharaf98
"""

# LAB - DEPENDENT INPUT :
# SCREEN W*H IN CM : taken from Lab_Equip_Setup_Summary doc:
# https://docs.google.com/spreadsheets/d/13x8n6MEI77dmya0CuO6wTysOv5k5ueiAgIYdlxZZ_Ec/edit?usp=sharing
SCREEN_SIZE = {'SA': np.array([78.7, 44.6]), 'SB': np.array([64, 36]), 'SC': np.array([69.8, 39.3]),
               'SD': np.array([58.5, 32.9]), 'SE': np.array([34.5, 19.5]), 'SF': np.array([34.5, 19.5]),
               'SX': np.array([53, 30]), 'SZ': np.array([53.5, 29.5])}
# VIEWING DISTANCE IN CM:
# taken from : https://docs.google.com/spreadsheets/d/13x8n6MEI77dmya0CuO6wTysOv5k5ueiAgIYdlxZZ_Ec/edit#gid=0
VIEWING_DIST = {'SA': 119, 'SB': 100, 'SC': 144, 'SD': 123, 'SE': 80, 'SF': 80, 'SX': 69.5, 'SZ': 71}

# PARAMETERS
STIM_VA = 2.3  # stimulus visual angle
STIM_DUR = 500  # stimulus duration in ms
ANALYZED_EYE = 'R'  # 'L' or 'R' for left or right eye: if subjects have binocular data, this eye will be the arbitrary choice # 2022-09-28 DMT following CONSORTIUM DECITION [Dejan Consult]
FIX_REF_ANGLE_RADIUS = 1.5  # Fixation stability was defined using a DIAMETER of 3° of visual angle; NOTE THIS IS THE RADIUS so 3/2 = 1.5 : decision made by LM in DMT
CENTER = 'center'
STIMULUS = 'stimulus'
FIX_RF_TYPE = [STIMULUS, CENTER]  # types of reference for fixation analysis: gaze w.r to stimulus / center

FACE = 'Face'
OBJ = 'Object'
BLANK = 'Blank'
# for replay:
FACE_TARGET = "Face_target"
OBJ_TARGET = "Obj_target"
FACE_NONT = "Face_non_target"
OBJ_NONT = "Obj_non_target"
STIM_TYPES = [FACE, OBJ]
STIM_LOCS = ['TopRight', 'TopLeft', 'BottomRight', 'BottomLeft']
GAME_WORLDS_LIST = ["World 1", "World 2", "World 3", "World 4"]
REPLAY_WORLDS_LIST = ["World A", "World B"]
FULL_WORLDS_LIST = GAME_WORLDS_LIST + REPLAY_WORLDS_LIST
STIM_PER_WORLD = 50  # how many stimuli are there in each world
FACE_PER_WORLD = 20
OBJ_PER_WORLD = 20
BLANK_PER_WORLD = 10
UNIQUE_FACE_STIM = 10
UNIQUE_OBJ_STIM = 10
VIS_LIST = ["True Positive", "False Negative", "False Positive", "True Negative"]
EPOCH_START = 1000  # time before stimulus onset that defines trial onset in MILLISECONDS
EPOCH_END = 2500  # time after stimulus onset that defines trial end in MILLISECONDS
PRE_STIM_DUR = 500  # time windows of pre-stimulus activity in MILLISECONDS

# blink padding parameter
BLINK_PAD_MS = 200

# eyelink message types
RECORDING_INFO = "RECCFG"
GAZE_COORDS = "GAZE_COORDS"

# full logs messages
SHOWING_STIMULUS = 'SHOWING_STIMULUS'
# the following mapping is based on column 9 in the full logs and its meaning according to the internal analyzer calculations
# in AnalyzerOutput (subs' BEH). It was verified using 3 full logs types: practice world log, game world log, replay world log.
STIM_LOCATION_MAP = {'BottomLeft': (0.3, 0.3), 'BottomRight': (0.7, 0.3),'TopLeft': (0.3, 0.7), 'TopRight': (0.7, 0.7)}

SCREEN_DIMS = "Screen Pixel dimensions :: "


def get_screen_dims(sess_info_path):
    """
    Gets the actual screen dimentions in pixels from the subject's session info file
    :param sess_info_path:
    :return: Width, Height
    """
    with open(sess_info_path) as f:
        sess_info = f.readlines()

    screen_dims_line = [sess_info[i] for i, s in enumerate(sess_info) if SCREEN_DIMS in s][0]
    screen_dims_line = screen_dims_line.replace(SCREEN_DIMS, "")
    screen_dims = screen_dims_line.split("x")
    screen_width_px = float(screen_dims[0])
    screen_height_px = float(screen_dims[1])
    return screen_width_px, screen_height_px


class Triggers:
    # a class to act as a struct object that will hold the ID information for stimuli, orientation, duration,
    # and task: https://twcf-arc.slab.com/posts/video-game-trigger-codes-7mtc2s2u
    def __init__(self):
        # stimuli
        self.Stimuli = [None] * 250
        # GAME STIMULI
        self.Stimuli[1:10] = [FACE] * UNIQUE_FACE_STIM  # faces
        self.Stimuli[21:30] = [OBJ] * UNIQUE_OBJ_STIM  # objects
        self.Stimuli[50] = BLANK  # blank
        # REPLAY STIMULI
        self.Stimuli[101:111] = [FACE_TARGET] * UNIQUE_FACE_STIM  # faces (FACE TARGET)
        self.Stimuli[121:131] = [OBJ_NONT] * UNIQUE_OBJ_STIM  # objects (FACE TARGET)
        self.Stimuli[150] = BLANK  # blank (FACE TARGET)
        self.Stimuli[201:211] = [FACE_NONT] * UNIQUE_FACE_STIM  # faces (OBJ TARGET)
        self.Stimuli[221:231] = [OBJ_TARGET] * UNIQUE_OBJ_STIM  # objects (OBJ TARGET)
        self.Stimuli[250] = BLANK  # blank (OBJ TARGET)

        # trial onsets (which according to our current scheme is the same as the stimulus category trigger)
        self.TrialOnset = np.hstack((np.arange(0, 10), np.arange(20, 30)))
        self.ProbeOnset = '100'
        # other codes
        self.Filler = {'95': 'Filler', '195': 'Filler'}  # 95 is game filler, 195 is replay filler
        self.AnimationPeakEnd = '253'  # this is when the stimulus begins shrinking/fading from full size
        # response: 196, 198 are replay options
        self.Response = {'96': 'NoAnswer', '98': 'Yes', '99': 'No'}
        self.ReplayResponse = {'196': 'NoResp', '198': 'Resp'}  # 196 actually denotes the end of the response-window in the replay

        # location
        self.Location = {'60': 'TopLeft', '70': 'TopRight', '80': 'BottomRight', '90': 'BottomLeft',
                             '61': 'TopLeft', '71': 'TopRight', '81': 'BottomRight', '91': 'BottomLeft'}

        # level begin/end
        self.LevelBegin = '251'
        self.LevelEnd = '252'


def find_tracked_eye(ascii_file_path):
    """
    Given a random ascii file path (it doesn't matter which file it is), find the line with the following information:
    'RECCFG CR 1000 2 0 ???' and extract the tracked eye. Options for tracked eye (in ???) are:
    - L : left eye
    - R: right eye
    - LR: both
    This is the tracked eye for this subject as we assume that we do not change the tracked eye for a single subject
    within-experiment.
    :param ascii_file_path:
    :return: the tracked eye in this ascii file.
    """
    random_ascii_file = open(ascii_file_path, 'r')
    file_content = random_ascii_file.read().splitlines(True)  # split into lines
    random_ascii_file.close()
    for line in file_content:
        if RECORDING_INFO in line:
            msg = line.split(" ")  # the information has both "\t" and " "  in it but to get what we want we use " "
            tracked_eye = msg[-1].replace("\n", "")  # as the L/R info is at the end of the line
            break
    return tracked_eye


def define_analyzed_eye(eye):
    if ANALYZED_EYE in eye:  # if ANALYZED_EYE was tracked
        return ANALYZED_EYE
    else:  # for labs who did not have tracking of the ANALYZED_EYE, we have no choice
        return eye


def InitParams(subject_name, sess_info_path, ascii_file_path):
    """
    defines and returns the parameters that will be used for analysis
    :return:
    """
    params = dict([])
    sub_lab = subject_name[:2]
    # lab dependent vars
    params['SubjectName'] = subject_name
    params['ScreenWidth'] = SCREEN_SIZE[sub_lab][0] * 10  # transformation from cm to mm
    params['ScreenHeight'] = SCREEN_SIZE[sub_lab][1] * 10  # transformation from cm to mm
    params['ViewDistance'] = VIEWING_DIST[sub_lab] * 10  # transformation from cm to mm
    params['ScreenResolution'] = np.array(get_screen_dims(sess_info_path))
    params['TrackedEye'] = find_tracked_eye(ascii_file_path)
    params['Eye'] = define_analyzed_eye(params['TrackedEye'])
    params['stimAng'] = STIM_VA
    params['Triggers'] = Triggers()
    return params


def UpdateParams(params, msgDF):  # fulllog_path
    """
    This function updates the parameters dictionary based on the eye tracker messages
    :param params: the parameters dict to be updated (should be the output of InitParams)
    :param eyeDFs: dataframe containing all ET messages
    :param fulllog_path: the path to one full log file where the stimulus information is
    :return: an updated parameter dictionary
    """
    # the sampling rate can be extracted from any Eyelink MSG line that contains RECORDING_INFO string
    # the structure of a line in msgDF containing RECORDING_INFO string is that in the text column its value is:
    # "<RECORDING_INFO> <tracking mode> <sample rate> <filter settings for the file data> <filter settings for the link data> <tracked eyes>"
    # (filter settings: 0 is off,1 is standard, and 2 is extra.)
    # for example: "RECCFG CR 1000 2 0 LR"

    recording_info = msgDF[msgDF["text"].str.contains(RECORDING_INFO)].iloc[0]['text'].split(" ")
    params['SamplingFrequency'] = float(recording_info[2])

    # get the screen resolution (assumed to be in the third message)  --> NOT NEEDED AS WE HAVE THAT INFO FROM THE LABS
    # get the screen resolution from the first Eyelink MSG line that contains GAZE_COORDS string
    # the structure of such a line is: "GAZE_COORDS <0 width> <0 height> <1 width> <1 height>"
    # for example: "GAZE_COORDS 0.00 0.00 1919.00 1079.00
    scMsg = msgDF[msgDF["text"].str.contains(GAZE_COORDS)].iloc[0]['text'].split(" ")

    # screen conversion factor (how many CM per pixel) : ScreenWidth is the actual width IN MILLIMIETERS (so /10)
    # and screen resolution[0] is the pixels in width
    params['PixelPitch'] = [(params['ScreenWidth']/10) / params['ScreenResolution'][0],
                            (params['ScreenHeight']/10) / params['ScreenResolution'][1]]

    # center location
    params['ScreenCenter'] = params['ScreenResolution'] / 2

    # view_distance in cm = view_distance / 10 as 'ViewDistance' is in MILLIMETERS
    params['stimPix'] = AnalysisHelpers.deg2pix(params['ViewDistance']/10, params['stimAng'], params['PixelPitch'])
    params['DegreesPerPix'] = params['stimAng'] / params['stimPix']
    params['StimulusCoords'] = GetStimCoords(params)  # fulllog_path

    return params


def GetStimCoords(params):  # fulllogs
    """
    This function extracts the coordinates (in pixels) of the stimuli in different locations from the fulllogs
    :param fulllogs: the path to a fulllogs file that contains the stimulus manager information
    :param params: the parameters dictionary (output of InitParams and UpdateParams)
    :return: stimCoords
    """
    # BASED ON KNOWN MAPPING FROM BEH FILES:
    res = dict()  # stimulus location coordination mapping between location name and its pixel location on the screen
    for loc in STIM_LOCATION_MAP.keys():
        loc_coords = STIM_LOCATION_MAP[loc]
        xloc = loc_coords[0] * params['ScreenResolution'][0]
        yloc = loc_coords[1] * params['ScreenResolution'][1]
        res[loc] = (xloc, yloc)

    return res
