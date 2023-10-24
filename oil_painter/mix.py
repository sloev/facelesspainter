from mido import MidiFile
import os
import random
from PIL import Image,ImageOps,ImageFilter,ImageEnhance
from collections import Counter
from itertools import cycle
from tqdm import tqdm
import json
import numpy as np
import time
import math 
from colorsys import rgb_to_hsv
img = Image.open("neon.jpeg")

from easing_functions import *


def load(name):
    mid = MidiFile(name+".mid")
    mididict = []
    output = []

    # Put all note on/off in midinote as dictionary.
    for i in mid:
        if i.type == 'note_on' or i.type == 'note_off' or i.type == 'time_signature':
            mididict.append(i.dict())
    # change time values from delta to relative time.
    mem1=0
    for i in mididict:
        time = i['time'] + mem1
        i['time'] = time
        mem1 = i['time']
    # make every note_on with 0 velocity note_off
        if i['type'] == 'note_on' and i['velocity'] == 0:
            i['type'] = 'note_off'
    # put note, starttime, stoptime, as nested list in a list. # format is [type, note, time, channel]
        if i['type'] == 'note_on':
            output.append({'instrument': name,'note':i['note'], 'time':i['time'], 'vel':i['velocity']})
    return output

clicks_cache = {}




midi_data = []
for n in ["snare"]:
    midi_data.extend(load(n))
midi_data = sorted(midi_data, key=lambda d: d["time"])
print(midi_data)


fps = 30.0
current_time = 0.0


last_event=None
decay_in_seconds = 3

frames = []

while True:
    events = []
    if midi_data:
        while midi_data:
            if midi_data[0]["time"] <= current_time:
                last_event =midi_data.pop(0)
                events.append(last_event)
            else:
                break
    else:
        if last_event["time"] + decay_in_seconds < current_time:
            break

    frames.append(events)

    current_time += 1.0/fps

harp_path = "./output_frames"
turing_path = "./turing_output_frames"
output_path = "./mix_output"

images = []
harp_images = [os.path.join(harp_path, a) for a in sorted(os.listdir(harp_path)) if a.endswith("jpg")]
turing_images = [os.path.join(turing_path, a) for a in sorted(os.listdir(turing_path)) if a.endswith("jpg")]
for i in range(max(len(harp_images), len(turing_images))):
    a=b=None
    try:
        a = harp_images.pop(0)
    except:pass
    try:
        b = turing_images.pop(0)
    except:pass

    images.append((a,b))


velocities = [a["vel"] for f in frames for a in f]
low = min(velocities)
high = max(velocities)
max_seconds_decay = 3.0
fade_start = 0
fader = None

skip = True
index = 0
iterations = 0

for i, frame_events in enumerate(tqdm(frames)):
    harp_image, turing_image = images.pop(0)

    for f in frame_events:
        
        if fader is None :
            index = 0

            iterations = ((f["vel"] - low)/(high-low)) * max_seconds_decay * fps
            fader = CubicEaseIn(start=1, end=0, duration=iterations)

        
    harp_image = Image.open(harp_image)


    
    background = harp_image.copy()
    if index>= iterations:
        fader = None
    if fader is not None and turing_image is not None:
        
        alpha = fader.ease(index)
        
        index +=1

        turing_image = Image.open(turing_image)
        # background = turing_image.copy()

        background = Image.blend(harp_image,turing_image, alpha)
        print("alpha", alpha)

    background.save(os.path.join(output_path, f'{i:08}.jpg'), quality=100)


    # print("turing_blend", turing_blend)











