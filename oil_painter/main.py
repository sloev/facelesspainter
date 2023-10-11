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
colors_orig = [
    (255,70,70),
    (0,255,121),
    (125,255,95),
    (0,236,255),
    (251,0,114),
    ( 235,255,0),
    (0,255,236)
]
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
                if not clicks:
                    print(f"skipping missing clicks: ./output/{prefix}.data.json")
                    continue
                clicks_cache.setdefault(prefix, cycle(clicks))
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
decay_in_seconds = 4

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
    prefix = next(prefix_gen)
    clicks = clicks_cache.get(prefix)
    image_gen = iterate_prefix(prefix)

    mask = Image.open(f"./output/{prefix}.mask.png").convert("L")
    
    c = next(clicks)

    x = int(c["x"])
    y = int(c["y"])
    # else:
    #     w,h = mask.size

    #     x,y = int(w/2+random.randint(-200,200)), int(h/2 + random.randint(-200,200))
    
   

# Crop the center of the image
   
    print(f"particle created with note:{note}, vel:{velocity}, seconds:{seconds}, color {color}")
    for i in range(int(seconds*fps)):

        new_width = 1920+(i*(note/12.7))
        new_height = 1080+(i*(note/12.7))
        left = x - (new_width/2)
        top = y - (new_height/2)
        right = x + (new_width/2)
        bottom = y + (new_height/2)
        mask = next(image_gen)
        if mask is None:
            break
        mask2 = mask.crop((left, top, right, bottom))
        mask2 = mask2.resize((1920, 1080), Image.LANCZOS)
        mask2 = mask2.filter(ImageFilter.GaussianBlur(max(0,i-(fps*(velocity/127*2)))))
        filter = ImageEnhance.Brightness(mask2)
        mask2 = filter.enhance(min(1.0, 1.2-i/velocity)).convert("L")

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











