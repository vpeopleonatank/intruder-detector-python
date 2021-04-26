#!/usr/bin/env python3
"""
* Copyright (c) 2018 Intel Corporation.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser

import aiohttp
import cv2
import numpy
import time
import collections
import json
import signal
import pathlib

from numpy.lib.function_base import copy
from inference import Network
import requests
from multiprocessing import Process
import threading


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
CONF_CANDIDATE_CONFIDENCE = 5
CODEC = 0x31637661

# Opencv windows per each row
CONF_WINDOW_COLUMNS = 2

# Global variables
model_xml = ''
model_bin = ''
conf_labels_file_path = ''
accepted_devices = ["CPU", "GPU", "HETERO:FPGA,CPU", "MYRIAD", "HDDL"]
video_caps = []
rtsp_mode = False
is_async_mode = True


# Event class to store the intruder details
class Event:
    def __init__(self, event_time=None, intruder=None, count=None, frame=None):
        self.time = event_time
        self.intruder = intruder
        self.count = count
        self.frame = frame


# VideoCap class to manage the input source
class VideoCap(threading.Thread):
    def __init__(self, vc, cam_name, cams, is_cam, is_rtsp=False):
        self.input_width = vc.get(3)
        self.input_height = vc.get(4)
        self.vc = vc
        self.cam_name = cam_name
        self.is_cam = is_cam
        self.no_of_labels = 0
        self.last_correct_count = []
        self.total_count = []
        self.current_count = []
        self.changed_count = []
        self.candidate_count = []
        self.candidate_confidence = []
        self.frame = None
        self.loop_frames = 0
        self.frame_count = 0
        self.events = []
        self.video_name = 'video{}.mp4'.format(cams)
        self.vw = None
        self.is_rtsp = is_rtsp

        if self.is_rtsp:
            assert self.vc.isOpened()

            # this lets the read() method block until there's a new frame
            self.cond = threading.Condition()

            # this allows us to stop the thread gracefully
            self.running = False

            # keeping the newest frame around
            self.frame = None

            # passing a sequence number allows read() to NOT block
            # if the currently available one is exactly the one you ask for
            self.latestnum = 0

            # this is just for demo purposes		
            self.callback = None
            
            super().__init__(name='FreshestFrame')
            self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.vc.release()

    def run(self):
        counter = 0
        while self.running:
            # block for fresh frame
            (self.rv, img) = self.vc.read()
            assert self.rv
            counter += 1

            # publish the frame
            with self.cond: # lock the condition for this operation
                self.frame = img if self.rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum+1
                if seqnumber < 1:
                    seqnumber = 1
                
                self.rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not self.rv:
                    return (self.rv, self.latestnum, self.frame)

            return (self.rv, self.latestnum, self.frame)

    def init(self, size):
        self.no_of_labels = size
        for i in range(size):
            self.last_correct_count.append(0)
            self.total_count.append(0)
            self.changed_count.append(False)
            self.current_count.append(0)
            self.candidate_count.append(0)
            self.candidate_confidence.append(0)

    def init_vw(self, h, w):
        self.vw = cv2.VideoWriter(os.path.join(OUTPUT_VIDEO_PATH, self.video_name), CODEC,
                                  self.vc.get(cv2.CAP_PROP_FPS), (w,h), True)
        if not self.vw.isOpened():
            return -1, self.video_name
        return 0, ''


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
    if args.loop:
        if args.loop == "True" or args.loop == "true":
            LOOP_VIDEO = True
        elif args.loop == "False" or args.loop == "false":
            LOOP_VIDEO = False
        else:
            print("Invalid input for -lp/--loop. Defaulting to LOOP_VIDEO = False")
            LOOP_VIDEO = False
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


def get_used_labels(req_labels):
    """
    Read the model's label file and get the position of labels required by the application

    :param req_labels: intruders to be detected in the input source
    :return status: 0 on success, negative value on failure
            labels: On success, list of labels present in model's label file
            used_labels: On success, list of bool values where true indicates that a label in labels list at that position is
                    used in the application
    """
    global conf_labels_file_path
    used_labels = []

    if conf_labels_file_path:
        labels = []
        with open(conf_labels_file_path, 'r') as label_file:
            if not label_file:
                return [-4, [], []]
            labels = [x.strip() for x in label_file]

        if not labels:
            return [-5, [], []]

        for label in labels:
            if label in req_labels:
                used_labels.append(True)
            else:
                used_labels.append(False)

        return [0, labels, used_labels]

    return [-6, [], []]


TELEGRAM_SCRETKEY = "1612311763:AAGV5bXiYSH1sLVrwz2S-4cnKghXchcha3g"
CHAT_ID = "-1001442912650"
RESQUEST_URL = f"https://api.telegram.org/bot{TELEGRAM_SCRETKEY}"

def send_contents_to_telegram(num_of_intruders, image):
    current_time = time.strftime("%H:%M:%S ngay %d-%m-%Y")
    # messages = f"Phát hiện {num_of_intruders} đối tượng vào lúc {current_time}"
    # messages = f"Phat hien {num_of_intruders} doi tuong vao luc {current_time}"
    messages = f"Phat hien doi tuong xam nhap vao luc {current_time}"
    cv2.putText(image, messages, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 5)
    # message_url = f"{RESQUEST_URL}/sendMessage?chat_id={CHAT_ID}&text={messages}"
    # _ = requests.request("POST", message_url)
    photo_url = f"{RESQUEST_URL}/sendPhoto?chat_id={CHAT_ID}"
    _, img_encoded = cv2.imencode('.jpg', image)
    files = [
        ('photo', ('', img_encoded.tobytes(), 'image/png'))
    ]

    _ = requests.request("POST", photo_url, files=files)


def get_input():
    """
    Parse the configuration file

    :return status: 0 on success, negative value on failure
            streams: On success, list of VideoCap containing configuration file data
            labels: On success, labels or intruder to be detected
    """
    global CONFIG_FILE
    global video_caps
    global rtsp_mode
    labels = []
    streams = []

    assert os.path.isfile(CONFIG_FILE), "{} file doesn't exist".format(CONFIG_FILE)
    config = json.loads(open(CONFIG_FILE).read())
    for id, item in enumerate(config['inputs']):
        for idx, video in enumerate(item['video']):
            cams = idx + 1
            cam_name = "Cam {}".format(idx)
            if video.isdigit():
                video_cap = VideoCap(cv2.VideoCapture(int(video)), cam_name, cams, is_cam=True)
            elif "rtsp" in video:
                rtsp_mode = True
                video_cap = VideoCap(cv2.VideoCapture(video), cam_name, cams, is_cam=True, is_rtsp=True)
            else:
                if os.path.isfile(video):
                    video_cap = VideoCap(cv2.VideoCapture(video), cam_name, cams, is_cam=False)
                else:
                    return [-8, [video]]
            video_caps.append(video_cap)
        labels = item['label']

    for video_cap in video_caps:
        if not video_cap.vc.isOpened():
            return [-9, [video_cap.cam_name]]

        video_cap.init(len(labels))
    return [0, labels]


def save_json():
    """
    Write the video results to json files

    :return status: 0 on success, negative value on failure
    """
    global video_caps
    global EVENT_FILE
    global DATA_FILE
    events = []
    if video_caps:
        events = video_caps[0].events
    total = 0
    event_json = open(EVENT_FILE, 'w')
    if not event_json:
        return -10

    data_json = open(DATA_FILE, 'w')
    if not data_json:
        return -11

    data_json.write("{\n\t\"video1\": {\n")
    event_json.write("{\n\t\"video1\": {\n")
    events_size = len(events) - 1
    if events:
        fps = video_caps[0].vc.get(cv2.CAP_PROP_FPS)
        for i in range(events_size):
            event_json.write("\t\t\"%d\":{\n" % (i))
            event_json.write("\t\t\t\"time\":\"%s\",\n" % events[i].time)
            event_json.write("\t\t\t\"content\":\"%s\",\n" % events[i].intruder)
            event_json.write("\t\t\t\"videoTime\":\"%d\"\n" % float(events[i].frame / fps))
            event_json.write("\t\t},\n")
            data_json.write("\t\t\"%d\": \"%d\",\n" % (float(events[i].frame / fps), events[i].count))
        event_json.write("\t\t\"%d\":{\n" % events_size)
        event_json.write("\t\t\t\"time\":\"%s\",\n" % events[events_size].time)
        event_json.write("\t\t\t\"content\":\"%s\",\n" % events[events_size].intruder)
        event_json.write("\t\t\t\"videoTime\":\"%d\"\n" % float(events[events_size].frame / fps))
        event_json.write("\t\t}\n")
        data_json.write("\t\t\"%d\": \"%d\"\n" % (float(events[events_size].frame / fps), events[events_size].count))
        total = events[events_size].count
    event_json.write("\t}\n")
    event_json.write("}")
    data_json.write("\t},\n")
    data_json.write("\t\"totals\":{\n")
    data_json.write("\t\t\"video1\": \"%d\"\n" % total)
    data_json.write("\t}\n")
    data_json.write("}")
    event_json.close()
    data_json.close()
    return 0


def arrange_windows():
    """
    Arranges the windows so that they are not overlapping

    :return: None
    """
    global CONF_WINDOW_COLUMNS
    global video_caps
    spacer = 470
    row_spacer = 250
    cols = 0
    rows = 0
    window_width = 768
    window_height = 432

    # Arrange log window
    cv2.namedWindow("Intruder Log", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Intruder Log", 0, 0)

    # Arrange video windows
    for idx in range(len(video_caps)):
        if cols == CONF_WINDOW_COLUMNS:
            rows += 1
            cols = 1
            cv2.namedWindow(video_caps[idx].cam_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(video_caps[idx].cam_name, window_width, window_height)
            cv2.moveWindow(video_caps[idx].cam_name, spacer * cols, row_spacer * rows)
        else:
            cols += 1
            cv2.namedWindow(video_caps[idx].cam_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(video_caps[idx].cam_name, window_width, window_height)
            cv2.moveWindow(video_caps[idx].cam_name, spacer * cols, row_spacer * rows)


# Signal handler
def signal_handler(sig, frame):
    global video_caps
    global EVENT_FILE
    global DATA_FILE
    if video_caps:
        ret = save_json()
        if ret != 0:
            if ret == -10:
                print("Could not create event JSON file " + EVENT_FILE + "!")
            elif ret == -11:
                print("Could not create data JSON file " + DATA_FILE + "!")

    clean_up()
    sys.exit(0)


def clean_up():
    """
    Destroys all the opencv windows and releases the objects of videoCapture and videoWriter
    """
    global video_caps
    cv2.destroyAllWindows()
    for video_cap in video_caps:
        if video_cap.vw:
            video_cap.vw.release()
        if video_cap.vc:
            video_cap.vc.release()


def intruder_detector():
    """
    Process the input source frame by frame and detects intruder, if any.

    :return status: 0 on success, negative value on failure
    """
    global CONF_CANDIDATE_CONFIDENCE
    global LOG_WIN_HEIGHT
    global LOG_WIN_WIDTH
    global CONFIG_FILE
    global video_caps
    global conf_labels_file_path
    global is_async_mode
    global UI
    global LOOP_VIDEO

    parse_args()
    ret = check_args()
    if ret != 0:
        return ret, ""

    if not os.path.isfile(CONFIG_FILE):
        return -12, ""

    if not os.path.isfile(conf_labels_file_path):
        return -13, ""

    # Creates subdirectory to save output snapshots
    pathlib.Path(os.getcwd() + '/output/').mkdir(parents=True, exist_ok=True)

    # Read the configuration file
    ret, req_labels = get_input()
    if ret != 0:
        return ret, req_labels[0]

    if not video_caps:
        return -14, ''

    # Get the labels that are used in the application
    ret, label_names, used_labels = get_used_labels(req_labels)
    if ret != 0:
        return ret, ''
    if True not in used_labels:
        return -15, ''

    # Init a rolling log to store events
    rolling_log_size = int((LOG_WIN_HEIGHT - 15) / 20)
    log_list = collections.deque(maxlen=rolling_log_size)

    # Open a file for intruder logs
    log_file = open(LOG_FILE_PATH, 'w')
    if not log_file:
        return -16, ''

    # Initializing VideoWriter for each source
    if UI and not LOOP_VIDEO:
        for video_cap in video_caps:
            ret, ret_value = video_cap.init_vw(int(video_cap.input_height), int(video_cap.input_width))
            if ret != 0:
                return ret, ret_value

    # Initialise the class
    infer_network = Network()
    # Load the network to IE plugin to get shape of input layer
    n, c, h, w = infer_network.load_model(model_xml, TARGET_DEVICE, 1, 1, 2, CPU_EXTENSION)[1]
    # Arrange windows so that they are not overlapping
    arrange_windows()

    min_fps = min([i.vc.get(cv2.CAP_PROP_FPS) for i in video_caps])
    signal.signal(signal.SIGINT, signal_handler, )
    no_more_data = [False] * len(video_caps)
    start_time = time.time()
    inf_time = 0
    next_request_id = 1
    cur_request_id = 0
    # Main loop starts here. Loop over all the video captures

    if is_async_mode:
        print("Application running in async mode...")
    else:
        print("Application running in sync mode...")

    cnt = 0
    prev_frame = None
    while True:
        for idx, video_cap in enumerate(video_caps):
            # Get a new frame
            vfps = int(round(video_cap.vc.get(cv2.CAP_PROP_FPS)))
            for i in range(0, int(round(vfps / min_fps))):
                if is_async_mode:
                    ret, video_cap.next_frame = video_cap.vc.read()
                else:
                    if rtsp_mode:
                        ret, cnt, video_cap.frame = video_cap.read(seqnumber=cnt+1)
                    else:
                        ret, video_cap.frame = video_cap.vc.read()
                video_cap.loop_frames += 1
                # If no new frame or error in reading a frame, exit the loop
                if not ret:
                    no_more_data[idx] = True
                    break
            if no_more_data[idx]:
                stream_end_frame = numpy.zeros((int(video_cap.input_height), int(video_cap.input_width), 1),
                                               dtype='uint8')
                stream_end_message = "Stream from {} has ended.".format(video_cap.cam_name)
                cv2.putText(stream_end_frame, stream_end_message, (int(video_cap.input_width/2) - 30,
                            int(video_cap.input_height/2) - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(video_cap.cam_name, stream_end_frame)
                continue
            for i in range(video_cap.no_of_labels):
                video_cap.current_count[i] = 0
                video_cap.changed_count[i] = False

            subtracted_objects_count = 0
            # Resize to expected size (in model .xml file)
            # Input frame is resized to infer resolution
            if is_async_mode:
                in_frame = cv2.resize(video_cap.next_frame, (w, h))
                in_frame = in_frame.transpose((2, 0, 1))
                in_frame = in_frame.reshape((n, c, h, w))

                # Start asynchronous inference for specified request.
                infer_network.exec_net(next_request_id, in_frame)
                video_cap.frame = video_cap.next_frame
                # Async enabled and only one video capture
                if len(video_caps) == 1:
                    videoCapResult = video_cap
                # Async enabled and more than one video capture
                else:
                    # Get previous index
                    videoCapResult = video_caps[idx - 1 if idx - 1 >= 0 else len(video_caps) - 1]

            else:
                in_frame = cv2.resize(video_cap.frame, (w, h))

                if prev_frame is None:
                    prev_frame = in_frame.copy()
                else:
                    diff = cv2.absdiff(in_frame, prev_frame)
                    prev_frame = in_frame.copy()
                    diff_frame = in_frame.copy()
                    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5,5), 0)
                    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                    dilated = cv2.dilate(thresh, None, iterations=3)
                    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) > 0:
                        for idx, contour in enumerate(contours):
                            (x, y, cw, ch) = cv2.boundingRect(contour)

                            area = cv2.contourArea(contour)
                            if area < 1000:
                                continue
                            subtracted_objects_count += 1
                            cv2.rectangle(diff_frame, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
                            cv2.putText(diff_frame, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 3)

                    cv2.imshow("feed", dilated)
                    cv2.imshow("diff", diff)
                    cv2.imshow("diff frame", diff_frame)

                videoCapResult = video_cap
                if subtracted_objects_count > 0:
                    in_frame = in_frame.transpose((2, 0, 1))
                    in_frame = in_frame.reshape((n, c, h, w))

                    # Start synchronous inference for specified request.
                    infer_network.exec_net(cur_request_id, in_frame)

                    inf_start = time.time()
                    # Wait for the result
                    if infer_network.wait(cur_request_id) == 0:
                        inf_time = time.time() - inf_start
                        # Results of the output layer of the network
                        res = infer_network.get_output(cur_request_id)
                        for obj in res[0][0]:
                            label = int(obj[1]) - 1
                            # Draw the bounding box around the object when the probability is more than specified threshold
                            if obj[2] > CONF_THRESHOLD_VALUE and used_labels[label]:
                                videoCapResult.current_count[label] += 1
                                xmin = int(obj[3] * videoCapResult.input_width)
                                ymin = int(obj[4] * videoCapResult.input_height)
                                xmax = int(obj[5] * videoCapResult.input_width)
                                ymax = int(obj[6] * videoCapResult.input_height)
                                # Draw bounding box around the intruder detected
                                cv2.rectangle(videoCapResult.frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4, 16)

                        for i in range(videoCapResult.no_of_labels):
                            # Track if there is any change in number of detected objects
                            if videoCapResult.candidate_count[i] == videoCapResult.current_count[i]:
                                videoCapResult.candidate_confidence[i] += 1
                            else:
                                videoCapResult.candidate_confidence[i] = 1
                                videoCapResult.candidate_count[i] = videoCapResult.current_count[i]

                            if videoCapResult.candidate_confidence[i] == CONF_CANDIDATE_CONFIDENCE:
                                videoCapResult.candidate_confidence[i] = 0
                                videoCapResult.changed_count[i] = True
                            else:
                                continue

                            if videoCapResult.current_count[i] > videoCapResult.last_correct_count[i]:
                                videoCapResult.total_count[i] += videoCapResult.current_count[i] - videoCapResult.last_correct_count[i]
                                det_objs = videoCapResult.current_count[i] - videoCapResult.last_correct_count[i]
                                total_count = sum(videoCapResult.total_count)
                                for det_obj in range(det_objs):
                                    current_time = time.strftime("%H:%M:%S")
                                    log = "{} - Intruder {} detected on {}".format(current_time, label_names[i],
                                                                                   videoCapResult.cam_name)
                                    log_list.append(log)
                                    log_file.write(log + "\n")
                                    event = Event(event_time=current_time, intruder=label_names[i], count=total_count,
                                                  frame=videoCapResult.frame_count)
                                    videoCapResult.events.append(event)

                                snapshot_name = "output/intruder_{}.png".format(total_count)
                                captured_image = numpy.array(videoCapResult.frame)
                                cv2.imwrite(snapshot_name, captured_image)
                                th = threading.Thread(target=send_contents_to_telegram,
                                                      args=(det_objs, captured_image))
                                th.start()
                                # p = Process(target=send_contents_to_telegram(total_count, videoCapResult.frame))
                                # p.start()
                            videoCapResult.last_correct_count[i] = videoCapResult.current_count[i]

                        # Create intruder log window, add logs to the frame and display it
                        log_window = numpy.zeros((LOG_WIN_HEIGHT, LOG_WIN_WIDTH, 1), dtype='uint8')
                        for i, log in enumerate(log_list):
                            cv2.putText(log_window, log, (10, 20 * i + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.imshow("Intruder Log", log_window)
                        videoCapResult.frame_count += 1

                # Video output
                if UI and not LOOP_VIDEO:
                    videoCapResult.vw.write(videoCapResult.frame)

                log_message = "Async mode is on." if is_async_mode else \
                    "Async mode is off."
                cv2.putText(videoCapResult.frame, log_message, (10, int(videoCapResult.input_height) - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 10, 10), 1)
                inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
                    "Inference time: {:.3f} ms".format(inf_time * 1000)
                cv2.putText(videoCapResult.frame, inf_time_message, (10, int(videoCapResult.input_height) - 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
                fps_time = time.time() - start_time
                fps_message = "FPS: {:.3f} fps".format(1/fps_time)
                cv2.putText(videoCapResult.frame, fps_message, (10, int(videoCapResult.input_height) - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

                # Display the video output
                cv2.imshow(videoCapResult.cam_name, videoCapResult.frame)

            start_time = time.time()

            # Loop video to mimic continuous input if LOOP_VIDEO flag is True
            if LOOP_VIDEO and not videoCapResult.is_cam:
                vfps = int(round(videoCapResult.vc.get(cv2.CAP_PROP_FPS)))
                # If a video capture has ended restart it
                if videoCapResult.loop_frames > videoCapResult.vc.get(cv2.CAP_PROP_FRAME_COUNT) - int(round(vfps / min_fps)):
                    videoCapResult.loop_frames = 0
                    videoCapResult.vc.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if is_async_mode:
                # Swap infer request IDs
                cur_request_id, next_request_id = next_request_id, cur_request_id

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('p'):
            # while 1:
            #     print('in while')
            #     if cv2.waitKey(0) == ord('c'):
            #         break
            cv2.waitKey()
        elif cv2.waitKey(1) == ord('c'):
            pass

        # if key == 9:
        #     is_async_mode = not is_async_mode
        #     print("Switched to {} mode".format("async" if is_async_mode else "sync"))

        if False not in no_more_data:
            break

    ret = save_json()
    if ret != 0:
        return ret, ''

    infer_network.clean()
    log_file.close()
    return 0, ''


if __name__ == '__main__':
    status, value = intruder_detector()

    if status == 0:
        print("Success!")
    elif status == -1:
        print("Could not open for write" + value + "!")
    elif status == -2:
        print("You need to specify the path to the .xml file!")
        print("Use -m MODEL or --model MODEL")
    elif status == -3:
        print("You need to specify the path to the labels file!")
        print("Use -l LABELS or --labels LABELS")
    elif status == -4:
        print("Error in opening labels file!")
    elif status == -5:
        print("No labels found in label file!")
    elif status == -6:
        print("Labels file not found!")
    elif status == -7:
        print("Error in opening Configuration file " + CONFIG_FILE + "!")
    elif status == -8:
        print("Could not find the video file " + value + "!")
    elif status == -9:
        print("\nCould not open " + value + " for reading!")
    elif status == -10:
        print("Could not create event JSON file " + EVENT_FILE + "!")
    elif status == -11:
        print("Could not create data JSON file " + DATA_FILE + "!")
    elif status == -12:
        print(CONFIG_FILE + " configuration file not found!")
    elif status == -13:
        print(conf_labels_file_path + " label file not found!")
    elif status == -14:
        print("No input source found in configuration file!")
    elif status == -15:
        print("Error: No labels currently in use. Please edit " + CONFIG_FILE+" file!")
    elif status == -16:
        print("Error in opening intruder log file!")
    elif status == -17:
        print("Could not find the device!")
    else:
        print("Unknown error occurred!")

    clean_up()

