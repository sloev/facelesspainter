from PIL import Image, ImageDraw,ImageFilter, ImageEnhance
import json
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import Delaunay
import random
import math
import os
import time
from tqdm import tqdm


def rotate_point(px, py, ox, oy, theta):
    return (
        math.cos(theta) * (px-ox) - math.sin(theta) * (py-oy) + ox,
        math.sin(theta) * (px-ox) + math.cos(theta) * (py-oy) + oy,
    )


def makeRectangle(l, w, theta, x,y):
    return [
        rotate_point(x-(l/2), y-(w/2), x,y, theta),
        rotate_point(x, y-(w/2), x,y, theta),
        rotate_point(x, y+(w/2), x,y, theta),
        rotate_point(x-(l/2), y+(w/2), x,y, theta),
        rotate_point(x-(l/2), y-(w/2), x,y, theta),
    ]

def draw(input_prefix, output_prefix):
    try:
        im = Image.open(f"{input_prefix}.mask.png").convert("L").convert("RGB")

    except OSError:
        return
    
    if os.path.isdir(output_prefix):
        print(f"skipping {output_prefix}")
        return
    try:
        os.makedirs(output_prefix)

    except:pass



    background = Image.new("RGB", im.size)
    with open(f"{input_prefix}.data.json") as f:
        data = json.load(f)

    weight = data["width"]/2.0
    print(im.size)


    offset = im.size[0]*0.5

    xs = []
    ys =[]

    for p in data["points"]:
        x,y = p
        # y = ((int(y)-10000))+offset
        # x = ((int(x)-10000))+offset
        print(x,y)

        xs.append(x)
        ys.append(y)

    x = np.array(xs)
    n = np.arange(x.shape[0]) 
    y = np.array(ys)


    x_spline = interp1d(n, x,kind='cubic')
    seconds = 5
    framesPerSecond = 30
    steps  = seconds * framesPerSecond
    n_ = np.linspace(n.min(), n.max(), steps)
    y_spline = interp1d(n, y,kind='cubic')

    x_ = list(x_spline(n_))
    y_ = list(y_spline(n_))
    # draw = ImageDraw.Draw(im)

    # for i,(x,y) in tqdm(enumerate(list(zip(x_,y_)))):
    #     draw.ellipse((x-weight, y-weight, x+weight, y+weight), fill=(255))
    # im.save(f'{output_prefix}/{i:05}.jpg', quality=95)
    # return
    angles = []
    last_x_=last_y_=None
    for x,y in reversed(list(zip(x_, y_))):
        if last_x_ == last_y_ == None:
            last_x_ = x
            last_y_ = y
            continue
        dist = math.sqrt( (last_x_ - x)**2 + (last_y_ - y)**2 )

        dx = x - last_x_

        # Difference in y coordinates
        dy = y - last_y_

        # Angle between p1 and p2 in radians
        angles.append((math.atan2(dy, dx)+math.radians(270),dist))
        last_x_ = x
        last_y_ = y


    angle=dist = None
    # mask = Image.new("L", background.size, 0)
    # mask = im.convert("L")
    last_p1 = last_p2 = None
    mask = Image.new("L", background.size, 0)

    for i,(x,y) in tqdm(enumerate(list(zip(x_,y_)))):
        w = weight
        if angles:
            angle,dist = angles.pop(-1)

        draw = ImageDraw.Draw(mask)
        # v = makeRectangle(w, dist,angle, x,y)
        this_p1, this_p2 = rotate_point(x-(w), y-dist*2, x,y,angle),rotate_point(x+(w), y-dist*2, x,y,angle)

        if not last_p1:
            last_p1, last_p2 = rotate_point(x-(w), y+dist/2.0, x,y,angle),rotate_point(x+(w), y+dist/2, x,y,angle)
        points = np.array([list(a) for a in [this_p1, this_p2, last_p2, last_p1]])
        tri = Delaunay(points)
        for t in tri.simplices:
            ps = [tuple(points[i]) for i in t]
            ps.append(ps[0])
            draw.polygon(ps, fill=(255))

        # draw.ellipse((x-w, y-w, x+w, y+w), fill=(255))

        # mask.save(f'{output_prefix}/mask{i:05}.jpg', quality=95)
        last_p1, last_p2 = this_p1, this_p2 
        # continue

        # continue
        maskblur = mask.filter(ImageFilter.GaussianBlur(15))
       

       
        # if (i<30):
        #     blur = 90-i*3
        #     mask = mask.filter(ImageFilter.GaussianBlur(blur))
        #     filter = ImageEnhance.Brightness(mask)
        #     mask = filter.enhance(i/30)
        # else:
        #     mask = mask.filter(ImageFilter.GaussianBlur(5))

        # filter = ImageEnhance.Brightness(background)
        # background = filter.enhance(0.95)
        background = Image.composite(im,background, maskblur)
        background.save(f'{output_prefix}/{i:05}.jpg', quality=95)


while True:
    fs = os.listdir("./output/")
    for f in fs:
        if f.endswith("mask.png"):
            prefix = f.replace(".mask.png", "")
            input_prefix = f"./output/{prefix}"
            output_prefix = f"./frames/{prefix}"
            draw(input_prefix, output_prefix)
    time.sleep(5)

        
