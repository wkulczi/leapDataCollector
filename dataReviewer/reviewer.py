import os
import tkinter
import tkinter as tk

import PIL.Image
import PIL.ImageTk


def unpickle(filename):
    import pickle
    with open(filename, 'rb') as fp:
        banana = pickle.load(fp)
    return banana


class tkData(tk.Frame):
    def __init__(self, window, is_pkl=False, dataFolder="mpPhotos", activeFile='X-0', width=400,
                 height=400, row=0,
                 column=0):
        super().__init__(window)
        self.window = window

        # gets from parent
        self.dataFolder = dataFolder
        self.activeFile = activeFile
        self.is_pkl = is_pkl

        # ui stuff
        self.photo = None
        self.height = height
        self.width = width
        self.canvas = tkinter.Canvas(window, width=self.width, height=self.height)
        self.canvas.grid(row=row, column=column, sticky=tk.NSEW)

        # returns
        self.loadData()

    # shows image
    def loadData(self):
        from os.path import exists
        path_to_file = f"../train/{self.dataFolder}/{self.activeFile}.{'pickle' if self.is_pkl else 'jpeg'}"

        if self.is_pkl:
            if exists(path_to_file):
                self.canvas.create_text(200, 200, font="Ubuntu",
                                        text="Loading data...")
                self.canvas.delete("all")

                # open pickle and tell me the size of the data
                jointData = unpickle(path_to_file)
                pickleContents = "length of file"

                if type(jointData) == list:
                    if len(jointData) > 1:  # leap joints have the coords straight up
                        pickleContents = f"leapJoints file containing {len(jointData)} joints"
                    elif len(jointData) == 1:  # mp joints are wrapped in NormalizedLandmarkList
                        pickleContents = f"mpJoints file containing {len(jointData[0].landmark)} joints"
                    else:
                        pickleContents = "broken file"
                else:
                    self.canvas.create_text(200, 200, font="Ubuntu",
                                            text="pickle broken?")

                self.canvas.create_text(200, 200, font="Ubuntu",
                                        text=pickleContents)
            else:
                self.canvas.create_text(200, 200, font="Ubuntu",
                                        text=f"{self.activeFile}.pickle not found")

        else:
            if exists(path_to_file):
                im = PIL.Image.open(path_to_file)
            else:
                im = PIL.Image.new('RGB', (400, 400), (125, 125, 125))
            self.photo = PIL.ImageTk.PhotoImage(image=im)
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    # updates current image
    def updateImage(self, activeLetter):
        self.activeFile = activeLetter
        self.loadData()

    # (from parent) deletes opened file image
    def deleteData(self):
        os.remove(
            f"../train/{self.dataFolder}/{self.activeFile}.{'pickle' if self.is_pkl else 'jpeg'}")

    # (from parent)
    def renameData(self):
        pass

    def rewriteImage(self, targetLetter, lastFileWithTargetLetterIndex):
        import shutil

        newTargetLetterIndex = lastFileWithTargetLetterIndex + 1
        src = f"../train/{self.dataFolder}/{self.activeFile}.{'pickle' if self.is_pkl else 'jpeg'}"
        dst = f"../train/{self.dataFolder}/{targetLetter}-{newTargetLetterIndex}.{'pickle' if self.is_pkl else 'jpeg'}"

        shutil.copy(src, dst)
        os.remove(src)
        pass


def checkErrors(fileDict):
    errorLog = []
    for dataFolderX in fileDict.keys():
        for dataFolderY in fileDict.keys():
            sumOfFilesInX = sum(fileDict[dataFolderX].values())
            sumOfFilesInY = sum(fileDict[dataFolderY].values())
            if sumOfFilesInX == sumOfFilesInY:
                pass
            else:
                errorLog.append(f"Err {dataFolderX}[{sumOfFilesInX}] X {dataFolderY}[{sumOfFilesInY}]")
    return errorLog


class App:
    def __init__(self, root):
        # logic
        self.commandMode = False  # when in command mode user can move sample from X to Y

        self.activeFile = 'X0'
        self.activeLetter = 'X'
        self.globalIndex = -1
        self.allFiles = []
        self.mainCounter = dict()
        self.allLetters = []

        self.fileDict = {"mpPhotos": [], "leapCropped": [], "leapJoints": [], "mpJoints": [], "leapPhotos": []}
        self.fileNames = {"mpPhotos": dict(), "leapCropped": dict(), "leapJoints": dict(), "mpJoints": dict(),
                          "leapPhotos": dict()}
        self.countFiles()
        unevenFoldersList = checkErrors(self.fileDict)
        if unevenFoldersList:
            print("uneven numbers of elements in training data")
            for element in unevenFoldersList:
                print(element)

        # ui here
        self.activeLetterContainer = None
        self.logger = None
        self.commandModeIndicator= None
        self.initAppVariables()

        self.leapCroppedCanvas = tkData(root, dataFolder="leapCropped", activeFile=self.activeFile, row=0, column=1)
        self.leapJointsCanvas = tkData(root, dataFolder="leapPhotos",
                                       activeFile=self.activeFile, row=0, column=0)
        self.mpJointsCanvas = tkData(root, dataFolder="mpPhotos",
                                     activeFile=self.activeFile, row=0, column=2)
        self.mpJointsPickles = tkData(root, is_pkl=True, dataFolder="mpJoints",
                                      activeFile=self.activeFile, row=1, column=2)
        self.leapJointsPickles = tkData(root, is_pkl=True, dataFolder="leapJoints",
                                        activeFile=self.activeFile, row=1, column=0)
        self.root = root
        self.initGui(root)
        self.updateUi()

        self.root.bind("<KeyPress>", self.onKeyPress)
        self.root.mainloop()

    def initGui(self, root):
        # root.resizable(True, True)
        for i in range(3):
            root.rowconfigure(i, weight=1)
            root.columnconfigure(i, weight=1)

        self.activeLetterContainer = tk.Label(root, text="X-1", anchor=tk.constants.CENTER)
        self.activeLetterContainer.config(font=("Ubuntu", 24, 'bold'))
        self.activeLetterContainer.grid(row=1, column=1, sticky=tk.NSEW)

        self.logger = tk.Label(root, text="", anchor=tk.constants.CENTER)
        self.logger.config(font=("Ubuntu", 12))
        self.logger.grid(row=2, column=1, sticky=tk.NSEW)

        self.commandModeIndicator = tk.Label(root, text="", anchor=tk.constants.CENTER)
        self.commandModeIndicator.config(font=("Ubuntu", 12))
        self.commandModeIndicator.grid(row=2, column=0, sticky=tk.NSEW)

    def updateUi(self):
        self.activeLetterContainer.config(text=f"{self.activeFile.upper()}")
        self.leapCroppedCanvas.updateImage(self.activeFile)
        self.leapJointsCanvas.updateImage(self.activeFile)
        self.mpJointsCanvas.updateImage(self.activeFile)
        self.mpJointsPickles.updateImage(self.activeFile)
        self.leapJointsPickles.updateImage(self.activeFile)

    def log(self, text):
        self.logger.config(text=text)

    def updateCommandModeIndicator(self):
        if self.commandMode:
            self.commandModeIndicator.config(text="COMMAND MODE")
        else:
            self.commandModeIndicator.config(text="")

    def onKeyPress(self, event: tk.Event):
        # functional buttons
        if event.keysym in ['Escape', 'space', 'Return', 'Control_L', 'Control_R', 'Delete']:
            if event.keysym == 'Escape':
                self.root.destroy()
                self.root.quit()
            if event.keysym in ['Control_L', 'Control_R']:
                if self.commandMode:
                    self.commandMode = False
                    self.updateCommandModeIndicator()
                else:
                    self.commandMode = True
                    self.updateCommandModeIndicator()
            if event.keysym == 'Delete':
                self.delClick()
        else:
            if event.keysym == 'Right':
                self.rightArrowClick()
                self.log("")
            elif event.keysym == 'Left':
                self.leftArrowClick()
                self.log("")
            else:
                import re
                pattern = re.compile("^[a-z]$")
                if pattern.match(event.char):
                    if event.char.upper() != self.activeFile.split('-')[0]:
                        self.log(f"Sample {self.activeFile} moved to {event.char.upper()} class.")
                        self.rewriteClick(event.char.upper())
                        self.commandMode = False
                        self.updateCommandModeIndicator()

    # LOGIC GOES HERE

    def initAppVariables(self):
        self.allLetters = sorted(self.fileDict[list(self.fileDict.keys())[0]])
        folderNameWithMaxData = max(self.fileDict, key=lambda key: sum(self.fileDict[key].values()))
        self.allFiles = self.fileNames[folderNameWithMaxData]
        self.activeFile = self.allFiles[0]
        self.globalIndex = 0

    def countFiles(self):
        from os import listdir
        from os.path import isfile, join
        from collections import Counter
        import natsort
        for dataDir in self.fileDict.keys():
            files = [f for f in listdir(f"../train/{dataDir}") if
                     isfile(join(f"../train/{dataDir}", f))]
            self.fileDict[dataDir] = dict(Counter([x[0].upper() for x in files]))
            files = natsort.natsorted(files)
            self.fileNames[dataDir] = [x.split(".")[0] for x in files]

    # when right arrow
    def rightArrowClick(self):
        newIndex = self.globalIndex + 1
        if newIndex > (len(self.allFiles) - 1):
            self.globalIndex = 0
            self.activeFile = self.allFiles[0]
        else:
            self.activeFile = self.allFiles[newIndex]
            self.globalIndex = newIndex
        self.updateUi()

    def leftArrowClick(self):
        newIndex = self.globalIndex - 1
        if newIndex < 0:
            self.globalIndex = len(self.allFiles) - 1
            self.activeFile = self.allFiles[self.globalIndex]
        else:
            self.globalIndex = newIndex
            self.activeFile = self.allFiles[self.globalIndex]
        self.updateUi()

    def delClick(self):
        self.log(f"Deleted {self.activeFile} sample.")
        fileToDelete = self.activeFile
        self.leapCroppedCanvas.deleteData()
        self.leapJointsCanvas.deleteData()
        self.mpJointsCanvas.deleteData()
        self.mpJointsPickles.deleteData()
        self.leapJointsPickles.deleteData()
        self.rightArrowClick()

        self.allFiles.remove(fileToDelete)

    def rewriteClick(self, targetLetter):
        fileToDelete = self.activeFile

        self.leapCroppedCanvas.rewriteImage(targetLetter, max([int(x.split('-')[1]) for x in
                                                               self.fileNames[self.leapCroppedCanvas.dataFolder] if
                                                               x.split('-')[0] == targetLetter]))
        self.leapJointsCanvas.rewriteImage(targetLetter, max([int(x.split('-')[1]) for x in
                                                              self.fileNames[self.leapJointsCanvas.dataFolder] if
                                                              x.split('-')[0] == targetLetter]))
        self.mpJointsCanvas.rewriteImage(targetLetter, max([int(x.split('-')[1]) for x in
                                                            self.fileNames[self.mpJointsCanvas.dataFolder] if
                                                            x.split('-')[0] == targetLetter]))
        self.mpJointsPickles.rewriteImage(targetLetter, max([int(x.split('-')[1]) for x in
                                                             self.fileNames[self.mpJointsPickles.dataFolder] if
                                                             x.split('-')[0] == targetLetter]))
        self.leapJointsPickles.rewriteImage(targetLetter, max([int(x.split('-')[1]) for x in
                                                               self.fileNames[self.leapJointsPickles.dataFolder] if
                                                               x.split('-')[0] == targetLetter]))
        self.rightArrowClick()
        self.globalIndex = self.globalIndex - 1
        self.allFiles.remove(fileToDelete)
        pass

    # LOGIC END


if __name__ == "__main__":
    main = tk.Tk()
    # main.configure(background='black')
    app = App(main)
