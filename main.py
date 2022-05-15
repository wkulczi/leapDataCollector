import threading
import tkinter
import tkinter as tk

import cv2
import mediapipe as mp
import numpy as np
import PIL.Image, PIL.ImageTk

import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# |-----------------------------------------------------------------|
# |--- current letter --|---------------------|---leap skeleton-----|
# |---------------letter counters-------------|-leap video cropped--|
# |------raw camera feed, no cropping---------|------buttons?-------|
# |-----------------------------------------------------------------|
import Leap
from cvutils import getImageBorders, getImageFixedHeight, drawJointPosOnCanvas
from leaputils import convert_distortion_maps, undistort, getPixelLocation, unpackLeapVector, getFingerJoints, \
    getRawJointLocation, LeapCamType
from mputils import normalizeLandmarksToPx, drawFromMpLandmarks, getMyLandmarkStyles

aslChars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
labelCounters = dict()

mp_hands = mp.solutions.hands


class LeapCapture:
    def __init__(self, width=400, height=400, leapCamType: LeapCamType = LeapCamType.RAW_IMG):

        self.width = width
        self.height = height

        self.ret: bool = False
        self.frame = None
        self.joint_data = None

        self.controller = Leap.Controller()
        self.controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

        ## leapMotion inits

        self.right_coeff = None
        self.right_coordinates = None
        self.maps_initialized = False

        self.cam_type = leapCamType

        # thread management
        self.running = True
        self.thread = threading.Thread(target=self.process)
        self.thread.start()

    def process(self):
        while self.running:
            self.controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

            leapFrame: Leap.Frame = self.controller.frame()
            rawMediaPipeCoords = []
            rawCoords = []

            if not leapFrame.images.is_empty:
                leftImage, mainImage = leapFrame.images[0], leapFrame.images[1]
                cropHandCanvas = np.zeros((400, 400, 3), dtype=np.uint8)
                mpJointCanvas = np.ones((400, 400, 3), dtype=np.uint8)
                jointCanvas = np.zeros((400, 400, 3), dtype=np.uint8)
                if leftImage.is_valid:
                    if not self.maps_initialized:
                        self.right_coordinates, self.right_coeff = convert_distortion_maps(mainImage)
                        self.maps_initialized = True

                    undistorted_main = undistort(leftImage, self.right_coordinates, self.right_coeff, self.width,
                                                 self.height)

                    if self.cam_type == LeapCamType.RAW_IMG:
                        self.ret = True
                        self.frame = undistorted_main
                        self.joint_data = None
                    else:
                        if not leapFrame.hands.is_empty:
                            colorCoords = []
                            # get wrists pos
                            for hand in leapFrame.hands:
                                colorCoords.append(
                                    {"color": (0, 202, 255),
                                     "coords": getPixelLocation(hand.wrist_position, mainImage)})
                                colorCoords.append(
                                    {"color": (185, 190, 255),
                                     "coords": getPixelLocation(hand.palm_position, mainImage)})
                                rawCoords.append(
                                    {"pointID": "wrist", "position": unpackLeapVector(hand.wrist_position)})
                                rawCoords.append({"pointID": "palm", "position": unpackLeapVector(hand.palm_position)})
                                for finger in hand.fingers:
                                    colorCoords = colorCoords + getFingerJoints(finger, mainImage, withMetacarpal=True,
                                                                                withColors=True)
                                    rawCoords = rawCoords + getRawJointLocation(finger, withMetacarpal=True)

                            coords = [i["coords"] for i in colorCoords]
                            topY, bottomY, startX, endX = getImageBorders(coords)
                            if self.cam_type == LeapCamType.CROPPED_HAND:
                                # crop image based on coords
                                cropHandCanvas = getImageFixedHeight(oldImg=undistorted_main[
                                                                            topY - 20: bottomY + 20,
                                                                            startX - 20: endX + 20
                                                                            ], newHeight=400, newWidth=400)
                                self.ret = True
                                self.frame = cropHandCanvas
                                self.joint_data = None

                            elif self.cam_type == LeapCamType.JOINT_CANVAS:
                                # draw on image and crop
                                jointCanvas = drawJointPosOnCanvas(jointCanvas, colorCoords)
                                jointCanvas = getImageFixedHeight(oldImg=jointCanvas[
                                                                         topY - 20: bottomY + 20,
                                                                         startX - 20: endX + 20
                                                                         ], newHeight=400, newWidth=400,
                                                                  fillVal=(0, 0, 0),
                                                                  isGrayscale=False)
                                self.ret = True
                                self.frame = jointCanvas
                                self.joint_data = rawCoords

                        else:
                            self.ret = True
                            self.joint_data = None
                            if self.cam_type == LeapCamType.CROPPED_HAND:
                                self.frame = cropHandCanvas
                            elif self.cam_type == LeapCamType.JOINT_CANVAS:
                                self.frame = jointCanvas

                        if self.cam_type == LeapCamType.MP_JOINTS:
                            with mp_hands.Hands(
                                    model_complexity=0,
                                    max_num_hands=1,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5
                            ) as hands:
                                undistorted_main = cv2.cvtColor(undistorted_main, cv2.COLOR_GRAY2RGB)
                                results = hands.process(undistorted_main)
                                undistorted_main = cv2.cvtColor(undistorted_main, cv2.COLOR_RGB2BGR) #todo delete later
                                if results.multi_hand_landmarks:
                                    for hand_landmarks in results.multi_hand_landmarks:
                                        pxNormalizedCoords = normalizeLandmarksToPx(hand_landmarks, self.width,
                                                                                    self.height)
                                        mpJointCanvas = drawFromMpLandmarks(mpJointCanvas, pxNormalizedCoords,
                                                                            getMyLandmarkStyles())

                                        jointLocations = [i for i in pxNormalizedCoords.values()]
                                        topY, bottomY, startX, endX = getImageBorders(jointLocations)
                                        mpJointCanvas = getImageFixedHeight(
                                            mpJointCanvas[topY - 20: bottomY + 20, startX - 20: endX + 20], 400, 400,
                                            fillVal=(0, 0, 0),
                                            isGrayscale=False)
                                        rawMediaPipeCoords.append(results.multi_hand_landmarks[0])
                            self.ret = True
                            self.frame = mpJointCanvas
                            self.joint_data = rawMediaPipeCoords

    def get_frame(self):
        return self.ret, self.frame, self.joint_data  # todo

    def __del__(self):
        self.stop_and_kill()

    def stop_and_kill(self):
        print("del")
        # stop thread
        if self.running:
            self.running = False
            self.thread.join()

    def getType(self):
        return self.cam_type


class tkCamera(tk.Frame):
    def __init__(self, window, width=400, height=400, fps=60, row=0, column=0, vid=None):
        super().__init__(window)

        self.window = window

        if not vid:
            self.vid = LeapCapture(leapCamType=LeapCamType.RAW_IMG)
        else:
            self.vid = vid
        self.frame = None
        self.image = None
        self.joint_data = None
        self.running = True
        self.update_frame()

        self.width = width
        self.height = height
        self.canvas = tkinter.Canvas(window, width=self.width, height=self.height)
        self.canvas.grid(row=row, column=column, sticky=tk.NSEW)

        self.fps = fps
        self.delay = int(1000 / self.fps)

    def data_snapshot(self):
        if len(self.frame):
            return self.frame, self.joint_data

    def snapshot(self):
        if self.image:
            self.image.save("frame.jpg")

    def update_frame(self):
        # try with one frame first
        ret, frames, joint_data = self.vid.get_frame()

        if ret and frames is not None:
            self.joint_data = joint_data
            self.frame = frames
            self.image = PIL.Image.fromarray(frames)
            self.photo = PIL.ImageTk.PhotoImage(image=self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        if self.running:
            self.window.after(int(1000 / 60), self.update_frame)

    def stop(self):
        self.running = False
        self.vid.stop_and_kill()

    def start(self):
        if not self.running:
            self.running = True
            self.update_frame()

class DataStore:
    def __init__(self):
        

class App:

    def __init__(self, root):
        self.currentLetter = None
        self.activeLetter = ''
        self.initGui(root)
        self.root = root
        self.rawCapture = LeapCapture(leapCamType=LeapCamType.RAW_IMG)
        self.rawLeapCam = tkCamera(self.root, row=2, column=0, vid=self.rawCapture)
        self.croppedHandCapture = LeapCapture(leapCamType=LeapCamType.CROPPED_HAND)
        self.croppedHandCaptureCam = tkCamera(self.root, row=2, column=1, vid=self.croppedHandCapture)
        self.mediaPipeCapture = LeapCapture(leapCamType=LeapCamType.MP_JOINTS)
        self.mediaPipeCaptureCam = tkCamera(self.root, row=0, column=2, vid=self.mediaPipeCapture)
        self.jointCapture = LeapCapture(leapCamType=LeapCamType.JOINT_CANVAS)
        self.jointCaptureCam = tkCamera(self.root, row=1, column=2, vid=self.jointCapture)
        self.camFeeds = [self.rawLeapCam, self.jointCaptureCam, self.croppedHandCaptureCam, self.mediaPipeCaptureCam]
        self.root.bind("<KeyPress>", self.onKeyPress)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def cams_snapshot(self):
        frame, joints = self.croppedHandCaptureCam.data_snapshot()
        image = PIL.Image.fromarray(frame)
        image.save("dupa.jpeg")
        # for cam in self.camFeeds:
        #     print("letter: ", self.activeLetter, cam.data_snapshot())

    def on_closing(self):
        for cam in self.camFeeds:
            cam.stop()
        self.root.destroy()

    def initGui(self, root: tk.Tk):
        # root.resizable(True, True)
        for i in range(3):
            root.rowconfigure(i, weight=1)
            root.columnconfigure(i, weight=1)

        currentLetterContainer = ttk.Frame(root)
        currentLetterContainer.grid(row=0, column=0, sticky=tk.NS)
        for i in range(3):
            currentLetterContainer.rowconfigure(i, weight=1)

        currentLetterLabel = ttk.Label(currentLetterContainer, text="Active letter:")
        currentLetterLabel.grid(row=0, column=0, sticky=tk.EW)
        self.currentLetter = ttk.Label(currentLetterContainer, bootstyle="info", text="A",
                                       anchor=CENTER)  # todo change to dynamic
        self.currentLetter.config(font=("Ubuntu", 36, 'bold'))
        self.currentLetter.grid(row=1, column=0, sticky=tk.EW)

        ttk.Button(root, text='2').grid(row=1, column=0, sticky=tk.NSEW)  # this one is empty

        # ttk.Button(root, text='3').grid(row=2, column=0, sticky=tk.NSEW)  # skeleton input here

        counterFrameContainer = ttk.Frame(root)
        counterFrameContainer.grid(row=1, column=0, columnspan=2, sticky=tk.NSEW)

        for i in range(4):
            counterFrameContainer.rowconfigure(i, weight=1)

        for i in range(9):
            counterFrameContainer.columnconfigure(i, weight=1)

        for index, x in enumerate(aslChars):
            charFrame = tk.Frame(counterFrameContainer)
            charFrame.grid(row=index // 9, column=index % 9, sticky=tk.NSEW)
            for j in range(3):
                charFrame.columnconfigure(j, weight=1)
            letterLabel = ttk.Label(charFrame, text=x + ": ")
            letterLabel.grid(row=0, column=0)
            labelCounter = ttk.Label(charFrame, text='0')
            labelCounter.grid(row=0, column=1, columnspan=2, sticky=tk.W)
            labelCounters[x] = labelCounter  # tutaj wszystkie labelcountery

        ttk.Button(root, text='6').grid(row=2, column=1, sticky=tk.NSEW)
        ttk.Button(root, text='7').grid(row=0, column=2, sticky=tk.NSEW)
        ttk.Button(root, text='8').grid(row=1, column=2, sticky=tk.NSEW)
        ttk.Button(root, text='9').grid(row=2, column=2, sticky=tk.NSEW)

        # ttk.Label(frm, text="hey").grid(column=0, row=0)
        # ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)

    def onKeyPress(self, event: tk.Event):
        # functional buttons
        if event.keysym in ['Escape', 'space', 'Return']:
            if event.keysym == 'Escape':
                for cam in self.camFeeds:
                    cam.stop()
                self.root.destroy()
                self.root.quit()

            if event.keysym in ['space', 'Return']:
                self.cams_snapshot()
                print("Save")
        else:
            if self.activeLetter != event.char.upper():
                if event.char.upper() in labelCounters.keys():
                    self.currentLetter.config(text=event.char.upper())
                    self.activeLetter = event.char.upper()
                    print('save', self.activeLetter)
            else:
                print("save as well", self.activeLetter)


main = tk.Tk()
style = ttk.Style("flatly")
app = App(main)
