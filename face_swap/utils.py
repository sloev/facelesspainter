import glob, os, re
import cv2
import random

def get_latest_index(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    currentImages = glob.glob(f"{folder}*.png")
    index = 0 
    for img in currentImages:
        i = os.path.splitext(img)[0]
        try:
            num = re.findall('[0-9]+$', i)[0]
            index += 1
        except IndexError:
            pass
    return index

def random_image_from_folder(folder):
    currentImages = glob.glob(f"{folder}*.png")
    return random.choice(currentImages)


def save_image(folder, image):
    """
    Save the current image to the working directory of the program.
    """
    if not os.path.isdir(folder):
        os.makedirs(folder)

    currentImages = glob.glob(f"{folder}*.png")
    numList = [0]
    for img in currentImages:
        i = os.path.splitext(img)[0]
        try:
            num = re.findall('[0-9]+$', i)[0]
            numList.append(int(num))
        except IndexError:
            pass
    numList = sorted(numList)
    newNum = numList[-1]+1
    saveName = folder + '%012d.png' % newNum

    cv2.imwrite(saveName, image)
