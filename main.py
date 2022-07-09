import threading
import tkinter
import tkinter as tk
from tkinter.constants import *

import cv2
import mediapipe as mp
import numpy as np
import PIL.Image, PIL.ImageTk

# |-----------------------------------------------------------------|
# |--- current letter --|---------------------|---leap skeleton-----|
# |---------------letter counters-------------|-leap video cropped--|
# |------raw camera feed, no cropping---------|------buttons?-------|
# |-----------------------------------------------------------------|
import Leap
from cvutils import getImageBorders, getImageFixedHeight, drawJointPosOnCanvas
from leaputils import convert_distortion_maps, undistort, getPixelLocation, unpackLeapVector, getFingerJoints, \
    getRawJointLocation, LeapCamType, ImportDataType
from mputils import normalizeLandmarksToPx, drawFromMpLandmarks, getMyLandmarkStyles

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

    def get_cam_type(self):
        return self.cam_type

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
                                    min_detection_confidence=0.3,
                                    min_tracking_confidence=0.5
                            ) as hands:
                                undistorted_main = cv2.cvtColor(undistorted_main, cv2.COLOR_GRAY2RGB)
                                results = hands.process(undistorted_main)
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
        self.camType = self.vid.get_cam_type()
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

    def get_type(self):
        return self.camType

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
        self.aslCounter = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0,
                           'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 0,
                           'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0, '0': 0,
                           '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}

        self.DATA_FOLDER = "train"
        self.HAND_CROPPED_FOLDER = "leapCropped"
        self.LEAP_JOINTS_PHOTOS_FOLDER = "leapPhotos"
        self.MP_PHOTOS_FOLDER = "mpPhotos"
        self.LEAP_JOINTS_FOLDER = "leapJoints"
        self.MP_JOINTS_FOLDER = "mpJoints"
        self.dataDirs = [self.HAND_CROPPED_FOLDER, self.LEAP_JOINTS_PHOTOS_FOLDER, self.MP_PHOTOS_FOLDER,
                         self.LEAP_JOINTS_FOLDER, self.MP_JOINTS_FOLDER]

    def getCounters(self):
        return self.aslCounter

    def initCounters(self):
        import os
        for dataDir in self.dataDirs:
            if not os.path.exists(f"{self.DATA_FOLDER}/{dataDir}"):
                os.makedirs(f"{self.DATA_FOLDER}/{dataDir}")

        fileCounters = self.countAllDataOccurrences()

        if not all(x == fileCounters[0] for x in fileCounters):
            print("FILE DISSIMILARITIES")  # debug here

        self.updateCounters(fileCounters)

    def updateCounters(self, fileCounters):
        for k, v in self.aslCounter.items():
            if k in fileCounters[0].keys():
                self.aslCounter[k] = fileCounters[0][k]

    def countAllDataOccurrences(self):
        from os import listdir
        from os.path import isfile, join
        from collections import Counter
        fileCounters = []
        for dataDir in self.dataDirs:
            files = [f for f in listdir(f"{self.DATA_FOLDER}/{dataDir}") if
                     isfile(join(f"{self.DATA_FOLDER}/{dataDir}", f))]
            fileCounters.append(dict(Counter([x[0].upper() for x in files])))
        return fileCounters

    def saveFiles(self, filesToSave, letter):
        for file in filesToSave:
            # train /           filetype depending on data_letter_number
            filePath = f"{self.DATA_FOLDER}/{self.dataDirs[file['type']]}/{letter.upper()}-{self.aslCounter[letter.upper()]}"
            if file["type"] > ImportDataType.MP_JOINT_CANVAS:
                import pickle
                with open(
                        filePath + '.pickle',
                        'wb') as f:
                    pickle.dump(file["joints"], f)
            else:
                file["photo"].save(f"{filePath}.jpeg")

        self.validateAndUpdateCounters(letter)

    def validateAndUpdateCounters(self, letter):
        fileCounters = self.countAllDataOccurrences()
        letterCounters = [x[letter] for x in fileCounters]
        if not all(x == letterCounters[0] for x in letterCounters):
            print(f"something went wrong when saving letter {letter}:\n"
                  f"|-{self.dataDirs[0]}-|-{self.dataDirs[1]}-|-{self.dataDirs[2]}-|-{self.dataDirs[3]}-|-{self.dataDirs[4]}-|\n"
                  f"|-{letterCounters[0]}-|-{letterCounters[1]}-|-{letterCounters[2]}-|-{letterCounters[3]}-|-{letterCounters[4]}-|")
        else:
            self.updateCounters(fileCounters)


class App:

    def __init__(self, root, dataStore: DataStore):
        if dataStore is None:
            self.dataStore = DataStore()
            self.dataStore.initCounters()
        else:
            self.dataStore = dataStore
        self.dataStore = dataStore
        self.currentLetter = None
        self.activeLetter = ''
        self.aslCounters = self.dataStore.getCounters()
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
        dataToSave = []
        for camFeed in self.camFeeds[1:]:
            snapshots = []
            frame, joints = camFeed.data_snapshot()
            image = PIL.Image.fromarray(frame)
            if camFeed.camType == LeapCamType.JOINT_CANVAS:
                snapshotData = dict()
                snapshotData["type"] = ImportDataType.LEAP_JOINTS
                snapshotData["joints"] = joints
                snapshots.append(snapshotData)
                snapshotData = dict()
                snapshotData["type"] = ImportDataType.JOINT_CANVAS
                snapshotData["photo"] = image
                snapshotData["joints"] = None
                snapshots.append(snapshotData)
            elif camFeed.camType == LeapCamType.MP_JOINTS:
                snapshotData = dict()
                snapshotData["type"] = ImportDataType.MP_JOINTS
                snapshotData["joints"] = joints
                snapshots.append(snapshotData)
                snapshotData = dict()
                snapshotData["type"] = ImportDataType.MP_JOINT_CANVAS
                snapshotData["photo"] = image
                snapshotData["joints"] = None
                snapshots.append(snapshotData)
            else:  # camFeed.camType == LeapCamType.CROPPED_HAND:
                snapshotData = dict()
                snapshotData["type"] = ImportDataType.CROPPED_HAND
                snapshotData["photo"] = image
                snapshots.append(snapshotData)
            dataToSave = dataToSave + snapshots

        self.dataStore.saveFiles(dataToSave, self.activeLetter)
        self.updateCounters(self.activeLetter)

    def updateCounters(self, checkedLetter):
        self.aslCounters = self.dataStore.getCounters()
        labelCounters[checkedLetter.upper()].configure(text=self.aslCounters[checkedLetter.upper()])
        print(f"Saved letter: {checkedLetter}: {self.aslCounters[checkedLetter.upper()]}", end="\t")
        print(f"Sum: {sum(self.aslCounters.values())}", end="\t")
        sampledLetters = ["A", "B","E","G","H","I","L","P","R","V","W"]
        print(f"Left: {(2000 * len(sampledLetters)) - sum(map(self.aslCounters.get, sampledLetters))}")


    def on_closing(self):
        for cam in self.camFeeds:
            cam.stop()
        self.root.destroy()

    def initGui(self, root: tk.Tk):
        # root.resizable(True, True)
        for i in range(3):
            root.rowconfigure(i, weight=1)
            root.columnconfigure(i, weight=1)

        currentLetterContainer = tk.Frame(root, background='black')
        currentLetterContainer.grid(row=0, column=0, sticky=tk.NS)
        for i in range(3):
            currentLetterContainer.rowconfigure(i, weight=1)

        currentLetterLabel = tk.Label(currentLetterContainer, text="Active letter:", foreground='white',
                                      background='black')
        currentLetterLabel.grid(row=0, column=0, sticky=tk.EW)
        self.currentLetter = tk.Label(currentLetterContainer, text="A", foreground='white', background='black',
                                      anchor=CENTER)  # todo change to dynamic
        self.currentLetter.config(font=("Ubuntu", 36, 'bold'))
        self.currentLetter.grid(row=1, column=0, sticky=tk.EW)

        counterFrameContainer = tk.Frame(root, background='black')
        counterFrameContainer.grid(row=1, column=0, columnspan=2, sticky=tk.NSEW)

        for i in range(4):
            counterFrameContainer.rowconfigure(i, weight=1)

        for i in range(9):
            counterFrameContainer.columnconfigure(i, weight=1)

        for index, x in enumerate(self.aslCounters.keys()):
            charFrame = tk.Frame(counterFrameContainer, background='black')
            charFrame.grid(row=index // 9, column=index % 9, sticky=tk.NSEW)
            for j in range(3):
                charFrame.columnconfigure(j, weight=1)
            letterLabel = tk.Label(charFrame, text=x + ": ", background='black', foreground='white')
            letterLabel.grid(row=0, column=0)
            labelCounter = tk.Label(charFrame, text=self.aslCounters[x], background='black', foreground='white')
            labelCounter.grid(row=0, column=1, columnspan=2, sticky=tk.W)
            labelCounters[x] = labelCounter  # tutaj wszystkie labelcountery

        # tk.Label(frm, text="hey").grid(column=0, row=0)
        # tk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)

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
                print("Save", end="\t")
        else:
            if self.activeLetter != event.char.upper():
                if event.char.upper() in labelCounters.keys():
                    self.currentLetter.config(text=event.char.upper())
                    self.activeLetter = event.char.upper()
                    # print('save', self.activeLetter)
            else:
                # print("save as well", self.activeLetter)
                pass


if __name__ == "__main__":
    dataStore = DataStore()
    dataStore.initCounters()
    main = tk.Tk()
    main.configure(background='black')
    app = App(main, dataStore)
