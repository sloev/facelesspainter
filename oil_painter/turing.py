from PIL import Image, ImageFilter, ImageDraw,ImageOps
import random
import math
import os
import numpy as np
from tqdm import tqdm


def get_white_noise_image(width, height):
    pil_map = Image.new("RGBA", (width, height), 255)
    random_grid = map(lambda x: (
            int(random.random() * 256),
            int(random.random() * 256),
            int(random.random() * 256)
        ), [0] * width * height)
    pil_map.putdata(list(random_grid))
    return pil_map.convert("RGB")

def rotate_point(px, py, ox, oy, theta):
    return (
        math.cos(theta) * (px-ox) - math.sin(theta) * (py-oy) + ox,
        math.sin(theta) * (px-ox) + math.cos(theta) * (py-oy) + oy,
    )
current_frame = None
background = None
combined_mask = None

def particle():
    global background,current_frame
    width, height = current_frame.size
    n = random.randint(100,min(width,height))
    stepsize = random.randint(5,20)

        
    max_width = int(width/3)
    max_height= int(height/3)

    w = random.randint(int(max_width/6), max_width)
    h = random.randint(int(max_height/6), max_height)
    x,y = random.randint(int(w/2), int(width-(w/2))), random.randint(int(h/2), int(height-(h/2)))
    
    angle = random.randint(0,360)
    theta = math.radians(angle)

    topleft, topright, bottomright, bottomleft = (
            rotate_point(x-w/2, y-(h/2), x,y,theta),
            rotate_point(x+w/2, y-(h/2), x,y,theta),
            rotate_point(x+w/2, y+(h/2), x,y,theta),
            rotate_point(x-w/2, y+(h/2), x,y,theta)
        )

    mask = Image.new("L", current_frame.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([topleft, topright, bottomright, bottomleft, topleft], fill=(255))

    xx,yy = 0,0
    # index = random.choice([0,1])
    index = 0
    for i in range(n):

        xx += int(math.cos(theta) * stepsize)
        yy += int(math.sin(theta) * stepsize)

        
        background.paste([current_frame,background][index], (xx, yy), mask)
        combineddraw = ImageDraw.Draw(combined_mask)
        combineddraw.polygon([topleft, topright, bottomright, bottomleft, topleft], fill=(0))

        yield mask
    yield None
        

    

def create_turing_pattern( radius=2, sharpen_percent=300):
    radius = random.randint(7,11)
    global background,current_frame

    color =     (255,70,70)

    background = background.convert('LA')
    rep = random.randint(20,80)

    for _ in tqdm(list(range(rep)), desc="turing"):
        background = background.filter(ImageFilter.BoxBlur(radius=radius))
        background = background.filter(ImageFilter.UnsharpMask(radius=radius, percent=sharpen_percent, threshold=0))
    
    background =  background.convert("L")

    background = ImageOps.colorize(background, black="black", white=color)
    background =  background.convert("RGB")



input_path = "./output_frames/"

output_path = "./turing_frames/"
particles = []
max_particles = 9
particles_chance = 0.7


for i,f in tqdm(list(enumerate(sorted(os.listdir(input_path))))):
    if f.rsplit(".", 1)[-1] not in ["jpg", "png"]:
        continue

    current_frame = Image.open(f"{input_path}{f}")

    if  background is None:
        background = current_frame.copy()

    background = Image.blend(background,current_frame, 0.5)#current_frame.copy()#Image.new("RGB", current_frame.size, 0)
    noisy = get_white_noise_image(background.width, background.height)
    background = Image.blend(background,noisy, 0.5)#current_frame.copy()#Image.new("RGB", current_frame.size, 0)

    create_turing_pattern()
    if combined_mask is not None:
        background.paste(current_frame, (0,0), combined_mask)

    particles_new = []
    combined_mask = Image.new("L", current_frame.size, 255)

    for p in tqdm(particles, desc="painting particle"):
        mask = next(p)
        if mask is None:
            continue

        particles_new.append(p)

    particles = particles_new
    if len(particles)<max_particles:# and random.random()>particles_chance:
        particles.append(particle())

    background.save(f"{output_path}{i:010d}.jpg", quality=100)
