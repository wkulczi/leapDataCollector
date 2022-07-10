import os
import tkinter
import tkinter as tk
import cv2

import PIL.Image
import PIL.ImageTk


class tkImage(tk.Frame):
    def __init__(self, window, dataFolder="leapCropped", activeFile='X-0', width=400,
                 height=400, row=0,
                 column=0):
        super().__init__(window)
        self.window = window

        # gets from parent
        self.dataFolder = dataFolder
        self.activeFile = activeFile

        # ui stuff
        self.photo = None

        self.contrast = 127
        self.brightness = 255
        self.cvimg = None
        self.original = None

        self.height = height
        self.width = width
        self.canvas = tkinter.Canvas(window, width=self.width, height=self.height)
        self.canvas.grid(row=row, column=column, sticky=tk.NSEW)

        # returns
        self.loadData()

    def apply_brightness_contrast(self, brightness=255, contrast=127, log=False):
        brightness = self.map(brightness, 0, 510, -255, 255)
        contrast = self.map(contrast, 0, 254, -127, 127)

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(self.original, alpha_b, self.original, 0, gamma_b)
        else:
            buf = self.original.copy()

        if contrast != 0:
            f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        if log:
            print(f"Brightness: {self.brightness} \t Contrast: {self.contrast}")
        # cv2.putText(buf, 'B:{},C:{}'.format(brightness, contrast), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
        #             2)
        return buf

    def map(self, x, in_min, in_max, out_min, out_max):
        return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

    def reloadChange(self):
        blue, green, red = cv2.split(self.cvimg)
        img = cv2.merge((red, green, blue))
        im = PIL.Image.fromarray(img)
        self.photo = PIL.ImageTk.PhotoImage(image=im)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    # shows image
    def loadData(self):
        from os.path import exists
        path_to_file = f"train/{self.dataFolder}/{self.activeFile}.jpeg"

        if exists(path_to_file):
            self.original = cv2.imread(path_to_file, 1)
            self.cvimg = self.original.copy()
            blue, green, red = cv2.split(self.cvimg)
            img = cv2.merge((red, green, blue))
            im = PIL.Image.fromarray(img)
            # im = PIL.Image.open(path_to_file)
        else:
            im = PIL.Image.new('RGB', (400, 400), (125, 125, 125))
        self.photo = PIL.ImageTk.PhotoImage(image=im)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        self.cvimg = self.apply_brightness_contrast(self.brightness, self.contrast)
        self.reloadChange()
        # updates current image

    def updateImage(self, activeLetter):
        self.contrast = 127
        self.brightness = 255
        self.activeFile = activeLetter
        self.loadData()

    def changeBrightness(self, increase=False):
        if increase:
            self.brightness = self.cap(self.brightness + 1, 0, 510)
        else:
            self.brightness = self.cap(self.brightness - 1, 0, 510)
        self.cvimg = self.apply_brightness_contrast(self.brightness, self.contrast, log=True)
        self.reloadChange()

    def changeContrast(self, increase=False):
        if increase:
            self.contrast = self.cap(self.contrast + 1, 0, 510)
        else:
            self.contrast = self.cap(self.contrast - 1, 0, 510)
        self.cvimg = self.apply_brightness_contrast(self.brightness, self.contrast, log=True)
        self.reloadChange()

    def overwriteImage(self):
        path_to_file = f"train/{self.dataFolder}/{self.activeFile}.jpeg"
        blue, green, red = cv2.split(self.cvimg)
        img = cv2.merge((red, green, blue))
        im = PIL.Image.fromarray(img)
        im.save(path_to_file)
        print(f"{self.activeFile} saved")

    @staticmethod
    def cap(val, min, max):
        if val > max:
            return max
        elif val < min:
            return min
        else:
            return val


class App:
    def __init__(self, root):
        self.activeFile = 'X0'
        self.activeLetter = 'X'
        self.globalIndex = -1
        self.allFiles = []

        self.countFiles()
        self.initAppVariables()

        # ui here
        self.leapCroppedCanvas = tkImage(root, dataFolder="leapCropped", activeFile=self.activeFile, row=0, column=0)
        self.activeLetterContainer = None

        self.root = root
        self.initGui(root)
        self.updateUi()

        self.root.bind("<KeyPress>", self.onKeyPress)
        self.root.mainloop()

    def countFiles(self):
        from os import listdir
        from os.path import isfile, join
        import natsort
        files = [f for f in listdir(f"train/leapCropped") if
                 isfile(join(f"train/leapCropped", f))]
        files = natsort.natsorted(files)
        self.allFiles = [x.split(".")[0] for x in files]

    def initAppVariables(self):
        self.activeFile = self.allFiles[0]
        self.globalIndex = 0

    def initGui(self, root):
        for i in range(1):
            root.rowconfigure(i, weight=1)
            root.columnconfigure(i, weight=1)

        self.activeLetterContainer = tk.Label(root, text="X-1", anchor=tk.constants.CENTER)
        self.activeLetterContainer.config(font=("Ubuntu", 24, 'bold'))
        self.activeLetterContainer.grid(row=1, column=0, sticky=tk.NSEW)

    def updateUi(self):
        self.activeLetterContainer.config(text=f"{self.activeFile.upper()}")
        self.leapCroppedCanvas.updateImage(self.activeFile)

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

    def changeGamma(self, increase=False):
        self.leapCroppedCanvas.changeBrightness(increase)

    def changeContrast(self, increase=False):
        self.leapCroppedCanvas.changeContrast(increase)

    def saveImage(self):
        self.leapCroppedCanvas.overwriteImage()

    def onKeyPress(self, event: tk.Event):
        if event.keysym == 'Escape':
            self.root.destroy()
            self.root.quit()
        elif event.keysym == 'Right':
            self.rightArrowClick()
        elif event.keysym == 'Left':
            self.leftArrowClick()
        elif event.keysym == "Return":
            self.leapCroppedCanvas.overwriteImage()
        else:
            if event.char.upper() in ["Q","E","A","D"]:
                if event.char.upper() in ["Q","E"]:
                    self.leapCroppedCanvas.changeContrast(event.char.upper() == "E") #if E = increase
                else:
                    self.leapCroppedCanvas.changeBrightness(event.char.upper() == "D") #if E = increase


if __name__ == "__main__":
    main = tk.Tk()
    app = App(main)
