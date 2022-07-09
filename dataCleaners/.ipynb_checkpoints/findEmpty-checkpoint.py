from os import listdir
from os.path import isfile, join
import mediapipe
from datetime import datetime
import natsort

def unpickle(filename):
    import pickle
    try:
        with open(filename, 'rb') as fp:
            banana = pickle.load(fp)
    except EOFError:
        banana = None
    return banana

def getEmptyJoints(folder):
    files = [f for f in listdir(f"../train/{folder}") if
             isfile(join(f"../train/{folder}", f))]
    files = natsort.natsorted(files)
    
    emptyFiles = []
    for file in files:
        buf = unpickle(f"../train/{folder}/{file}")
        if buf is None:
            emptyFiles.append(f"{file.split('.')[0]}.*")
    return emptyFiles

emptyLeap = getEmptyJoints("leapJoints")
emptyMp = getEmptyJoints("mpJoints")
emptyFiles = list(set(emptyLeap + emptyMp))

now = datetime.now()
dt = now.strftime("%H%M%S-%d%m")
with open(f"toDelete_{dt}.txt", "w") as textfile:
    for element in emptyFiles:
        textfile.write(element+"\n")

print(f"[{now.strftime('%H:%M:%S | %d.%m.%y')}] found and saved names of {len(emptyFiles)} empty pickles")