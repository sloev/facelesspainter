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



"""FIND UD AF HVOR PARTIKEL STARTER; HVIS UDEN FOR FRAMEN; DA ZOOM IND SÅ NOTE_ONSET SES BEDRE
"""
color_from_hex = lambda h: tuple(int(h[i:i+2].lower(), 16) for i in (1, 3, 5))
colors_orig = [
    "#e6194B",
    "#bfef45",
    "#f58231",
    "#42d4f4",
    "#FEFEFF",
    "#f032e6",
    "#bfef45",
    "#FF4B1F",
    "#1FFFAD",
    "#F8FF1F",
    "#1FFF1F",
    "#FF70A9",
    "#70DEFF",
    "#FF4775",
    "#D9FF70",
    "#89009E",
    "#FFD1A8",
    "#469990"
    "#FAD6FF"
]
colors_orig = [color_from_hex(h) for h in colors_orig]

def col_dist(rgb1,rgb2):
    return abs(rgb_to_hsv(*rgb1)[0]-rgb_to_hsv(*rgb2)[0])

colors = cycle(colors_orig)

# while colors_orig:    
#     if not colors:
#         c = colors_orig.pop(0)
#         colors.append(c)
#         continue
#     d = [(col_dist(colors[-1], c), i,c) for i,c in enumerate(colors_orig)]
#     d = sorted(d, key=lambda t: t[0], reverse=True)
#     colors.append(colors_orig.pop(d[0][1]))
    
def getColor(note):
  return next(colors)
#   in_min = 0
#   in_max = 127
#   out_min = 0
#   out_max = len(colors)
#   index =  (note - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
#   return colors[int(index)]


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
def get_images():
    while True:
        fs = os.listdir("./frames/")

        ds = [d for d in fs if os.path.isdir(f"./frames/{d}/")]

        random.shuffle(ds)
        for prefix in ds:
            print(f"opening ./output/{prefix}.data.json")
            with open(f"./output/{prefix}.data.json") as f:
                d = json.load(f)
                clicks = d.get("clicks")
                start_point = d.get("points")[0]
                if not clicks:
                    print(f"skipping missing clicks: ./output/{prefix}.data.json")
                    continue
                clicks_cache.setdefault(prefix, [cycle(clicks),start_point])
                yield prefix 
        time.sleep(5)


prefix_gen = get_images()

midi_data = []
for n in ["string", "chords"]:
    midi_data.extend(load(n))
midi_data = sorted(midi_data, key=lambda d: d["time"])


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


def iterate_prefix(prefix):
    for f in sorted(os.listdir(f"./frames/{prefix}/")):
        print(f)
        if f.endswith(".jpg"):
            yield Image.open(f"./frames/{prefix}/{f}").convert("L")
    yield None


def particle(note, velocity):
    seconds = (decay_in_seconds/127)*velocity
    color = getColor(note)
    color = (255,255,255)
    prefix = next(prefix_gen)
    clicks, start_point = clicks_cache.get(prefix)
       
    c = next(clicks)

    x = int(c["x"])
    y = int(c["y"])
    start_x, start_y = start_point
    mid_point_x, midpoint_y = (start_x +x)/2.0, (start_y+y)/2.0
    max_dist = max(abs(start_x-x), abs(start_y-y)) * 1.3
    sel_height = max(max_dist, 1080)
    sel_width = (sel_height / 9.0)*16.0
    
    image_gen = iterate_prefix(prefix)

    mask = Image.open(f"./output/{prefix}.mask.png").convert("L")
 
    # else:
    #     w,h = mask.size

    #     x,y = int(w/2+random.randint(-200,200)), int(h/2 + random.randint(-200,200))
    
   

# Crop the center of the image
    take_images = 5
    for i in range(take_images):
        next(image_gen)

   
    print(f"particle created with note:{note}, vel:{velocity}, seconds:{seconds}, color {color}")
    start_fade = int(seconds*fps*2.0/3.0)
    start_blur = int(seconds*fps*4.0/5.0)
    fade_duration = (int(seconds*fps)-start_fade)
    blur_duration = (int(seconds*fps)-start_fade)
    fader = CubicEaseIn(start=1, end=0, duration=fade_duration)
    blurer = CubicEaseIn(start=0, end=30, duration=blur_duration)
    mask = None
    for i in range(int(seconds*fps)):
        new_width = sel_width+(i*(note/12.7))
        new_height = sel_height+(i*(note/12.7))
        left = mid_point_x - (new_width/2)
        top = midpoint_y - (new_height/2)
        right = mid_point_x + (new_width/2)
        bottom = midpoint_y + (new_height/2)
        try:
            new_mask = next(image_gen)
            if new_mask is not None:
                mask = new_mask
        except StopIteration:
            pass    
        mask2 = mask.crop((left, top, right, bottom))
        mask2 = mask2.resize((1920, 1080), Image.LANCZOS)
        if i > start_blur:
            mask2 = mask2.filter(ImageFilter.GaussianBlur(blurer.ease(i-start_blur)))
        if i > start_fade:
            filter = ImageEnhance.Brightness(mask2)
            mask2 = filter.enhance(fader.ease(i-start_fade)).convert("L")

        image2 = ImageOps.colorize(mask2, black="black", white=color)

        yield image2, mask2

    yield


particles = []
for i, frame_events in enumerate(tqdm(frames)):
    for f in frame_events:
        particles.append(particle(f["note"], f["vel"]))
    new_particles =[]
    
    background = Image.new("RGB", (1920,1080))


    for p in particles:
        data = next(p)
        if data is None:
            continue

        image, mask = data
        
        background = Image.composite(image, background, mask)

        new_particles.append(p)
    particles = new_particles

    background.save(f'output_frames/{i:08}.jpg', quality=100)











