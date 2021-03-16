from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import numpy
import time
import collections
import json
import signal
import pathlib
from inference import Network

# CONSTANTS
CONFIG_FILE = "../resources/config.json"
EVENT_FILE = "../UI/resources/video_data/events.json"
DATA_FILE = "../UI/resources/video_data/data.json"
TARGET_DEVICE = "CPU"
OUTPUT_VIDEO_PATH = "../UI/resources/videos"
CPU_EXTENSION = ""
LOOP_VIDEO = False
UI = False
CONF_THRESHOLD_VALUE = 0.55
LOG_FILE_PATH = "./intruders.log"
LOG_WIN_HEIGHT = 432
LOG_WIN_WIDTH = 410
CONF_CANDIDATE_CONFIDENCE = 4
CODEC = 0x31637661

# Opencv windows per each row
CONF_WINDOW_COLUMNS = 2

# Global variables
model_xml = ''
model_bin = ''
conf_labels_file_path = ''
accepted_devices = ["CPU", "GPU", "HETERO:FPGA,CPU", "MYRIAD", "HDDL"]
video_caps = []
is_async_mode = True

def parse_args():
    """
    Parse the command line argument

    :return None:
    """
    global LOOP_VIDEO
    global TARGET_DEVICE
    global conf_labels_file_path
    global model_xml
    global model_bin
    global UI
    global CPU_EXTENSION
    global is_async_mode

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model's weights.",
                        required=True, type=str)
    parser.add_argument("-lb", "--labels", help="Labels mapping file", default=None,
                        type=str, required=True)
    parser.add_argument("-d", "--device", help="Device to run the inference (CPU, GPU, MYRIAD, FPGA or HDDL only)."
                                               "To run with multiple devices use MULTI:<device1>,<device2>,etc. "
                                               "Default option is CPU.",
                        required=False, type=str)
    parser.add_argument("-lp", "--loop", help="Loop video to mimic continuous input.", type=str, default=None)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-f", "--flag", help="sync or async", default="async", type=str)
    parser.add_argument("-ui", "--user_interface", help="User interface for the video samples", default="False", type=str)
    args = parser.parse_args()
    if args.model:
        model_xml = args.model
    if args.labels:
        conf_labels_file_path = args.labels
    if args.device:
        TARGET_DEVICE = args.device
    if args.flag == "sync":
        is_async_mode = False
    else:
        is_async_mode = True
    if args.user_interface:
        if args.user_interface == "True" or args.user_interface == "true":
            UI = True
        elif args.user_interface == "False" or args.user_interface == "false":
            UI = False
        else:
            print("Invalid input for -ui/--user_interface. Defaulting to UI = False")
            UI = False

    if args.cpu_extension:
        CPU_EXTENSION = args.cpu_extension


def check_args():
    """
    Validate the command line arguments
    :return status code: 0 on success, negative value on failure
    """
    global model_xml
    global conf_labels_file_path
    global TARGET_DEVICE


    if model_xml == '':
        return -2

    if conf_labels_file_path == '':
        return -3

    if 'MULTI' not in TARGET_DEVICE and TARGET_DEVICE not in accepted_devices:
        print("Unsupported device: " + TARGET_DEVICE)
        return -17
    elif 'MULTI' in TARGET_DEVICE:
        target_devices = TARGET_DEVICE.split(':')[1].split(',')
        for multi_device in target_devices:
            if multi_device not in accepted_devices:
                print("Unsupported device: " + TARGET_DEVICE)
                return -17
    return 0

def main():
    pass

if __name__ == '__main__':
    main()