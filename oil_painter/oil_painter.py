from skimage.transform import PiecewiseAffineTransform, warp
from skimage.io import imread, imsave
import math
import random
import cv2
from scipy.interpolate import griddata
import numpy as np


def get_points(height, width, shape, weight=100):
    points = [(height,width)]
    angle = 90

    src = []
    n = 30
    for i in range(1, int(shape[0]/n)):
        y,x = points[i-1]
        
        x_n = x + (math.cos(math.radians(angle)) * n)
        y_n = y + (math.sin(math.radians(angle)) * n)
        points.append((y_n, x_n))
        angle += random.randint(-20,20)
        src.extend([[(i - 1)*n + points[0][0], 0 + points[0][1]], [(i - 1)*n + points[0][0], shape[1] + points[0][1]]])
        #print('coor', src[-1])

    vertices = []
    faces = []
    last_x, last_y = None, None

    depth = 0.0

    for i in range(1, len(points)-1):
        y_1, x_1 = points[i-1]
        y_2, x_2 = points[i]

        angle_1_2 = math.atan2(
            y_2 - y_1, x_2 - x_1
        )
        width_left = weight#random.uniform(width * 0.98 , width * 1.02)
        width_right = weight#random.uniform(width * 0.98 , width * 1.02)

        x_1_l = x_1 + (math.cos(angle_1_2 - (math.pi / 2)) * width_left)
        x_1_r = x_1 + (math.cos(angle_1_2 + (math.pi / 2)) * width_right)

        y_1_l = y_1 + (math.sin(angle_1_2 - (math.pi / 2)) * width_left)
        y_1_r = y_1 + (math.sin(angle_1_2 + (math.pi / 2)) * width_right)

        # vertices.extend([
        #     [x_1_l, y_1_l, depth],
        #     [x_1, y_1, depth],
        #     [x_1_r, y_1_r, depth]
        # ])
        vertices.extend([[y_1_l, x_1_l], [y_1_r, x_1_r]])


   # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    #mesh.show()
    return src[:len(vertices)], vertices


def draw_points(r_x, r_y, background):
    height, width, _ = background.shape
    
    brush = random.randint(1,4)
    filename = f"brushes/brush_{brush}.jpg"
    image = imread(filename)
    #image[image[:,:,3] > 0][3] = 255
    image_border = cv2.copyMakeBorder(image, int(height), int(height), int(height), int(height), cv2.BORDER_CONSTANT) 
    
    src, dst = get_points(r_y, r_x, image.shape)
    print(len(src), len(dst))

    source = np.array([[int(y), int(x)] for y,x in src],dtype=np.int32)

    destination = np.array([[int(y), int(x)] for y,x in dst],dtype=np.int32)
    #print(src.shape, dst.shape)
    grid_x, grid_y = np.mgrid[0:height*2, 0:width*2]
    grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(height*2,width*2)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(height*2,width*2)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')
    warped_image = cv2.remap(image_border, map_x_32, map_y_32, cv2.INTER_CUBIC)
    #print(warped_image)
    imsave("lol.png", warped_image)

    left, right = [], []
    for index, (y,x) in enumerate(destination):
        if index % 2 == 0:
            right.append([x,y])
        else:
            left.append([x,y])
    dst = np.array(left + list(reversed(right)))
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.fillPoly(mask, [dst], (255, 255, 255))
    kernel = np.ones((5,5), np.uint8) 

    mask = cv2.dilate(mask, kernel, iterations=2) 
    #mask = cv2.erode(mask, kernel, iterations=2) 
    #rgba = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGBA)


    #warped_image = cv2.bitwise_and(warped_image, mask)
    #print('warped', warped_image.shape)
    #print(warped_image)
    
    #rgba = cv2.cvtColor(warped_image, cv2.COLOR_RGB2RGBA)
    #mask = np.where(mask==0, 255, 0).astype('uint8')
    #print(mask)
    #rgba[:,:,3] = mask[:,:,2]

 #   r,g,b = cv2.split(warped_image) 
 #   rgba = cv2.merge((r,g,b, mask[:,:,0]))


    return warped_image, int(height), int(height)

def draw_now(x,y,background, r=None,g=None,b=None, hex_color=None):
    if hex_color is not None:
        r,g,b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    height, width, _ = background.shape
    #x = random.randint(200, width-200)
    #y = random.randint(200, height-200)
    img, offset_y, offset_x = draw_points(5000,5000, background)
    img2 = img[offset_y - y:(offset_y - y)+height, offset_x - x:(offset_x - x)+width, :]
    #result = np.zeros((height, width, 4), np.uint8)

    print(x,y,)
    print(img2.shape, background.shape)
    #back = np.zeros((height, width, 4), dtype=np.uint8)
    #plt.imshow(img2)
    #imsave("back.png", background)

    

    background = blend_transparent(r,g,b,background, img2)
    imsave("new_back.png", background)
    #back[y-offset_y:y-offset_y+img.shape[0], x-offset_x:x-offset_x+img.shape[1]] = img
    

    #background = cv2.bitwise_or(warped_image, back)
    return background

def blend_transparent(r,g,b,face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    hsv = cv2.cvtColor(overlay_t_img, cv2.COLOR_BGR2HSV)
    overlay_img = np.zeros(overlay_t_img.shape, dtype=np.uint8)
    overlay_img[:,:,:] = (r,g,b)
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2HSV)
    overlay_img[:,:,2] = hsv[:,:,2]
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_HSV2BGR)


    #overlay_img = overlay_t_img#[:,:,:] # Grab the BRG planes
    overlay_mask = hsv[:,:, 2:] #overlay_t_img[:,:,3:]  # And the alpha plane
    overlay_mask = overlay_mask.astype('float32')

    # another option is to use a factor value > 1:
    overlay_mask *= 1.5

    # clip pixel intensity to be in range [0, 255]
    overlay_mask = np.clip(overlay_mask, 0, 255)

    # change type back to 'uint8'
    overlay_mask = overlay_mask.astype('uint8')
                                       
    print('overlay', overlay_mask.shape, flush=True)
    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

background = np.zeros((5000, 5000, 3), dtype=np.uint8)

colors = [
    '73D2DE',
    '218380',
    '8F2D56',
    'D81159',
    'FFBC42'
]
for i in range(5):
    background = draw_now(2500,2500,background,hex_color=random.choice(colors))
exit(1)

# height, width = 5000, 5000

# image = imread('brush_3.jpg')
# image_border = cv2.copyMakeBorder(image, int(height/2), int(height/2), int(height/2), int(width/2), cv2.BORDER_CONSTANT) 

# src, dst = foo(int(height/2), int(width/2), image.shape)
# print(len(src), len(dst))
# source = np.array([[int(y), int(x)] for y,x in src],dtype=np.int32)

# destination = np.array([[int(y), int(x)] for y,x in dst],dtype=np.int32)
# #print(src.shape, dst.shape)
# grid_x, grid_y = np.mgrid[0:height, 0:width]
# grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
# map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(height,width)
# map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(height,width)
# map_x_32 = map_x.astype('float32')
# map_y_32 = map_y.astype('float32')
# warped_image = cv2.remap(image_border, map_x_32, map_y_32, cv2.INTER_CUBIC)

# left, right = [], []
# for index, (y,x) in enumerate(destination):
#     if index % 2 == 0:
#         right.append([x,y])
#     else:
#         left.append([x,y])
# dst = np.array(left + list(reversed(right)))
# mask = np.zeros((height, width, 3), dtype=np.uint8)
# cv2.fillPoly(mask, [dst], (255,255,255))
# kernel = np.ones((5,5), np.uint8) 

# mask = cv2.dilate(mask, kernel, iterations=2) 
# #mask = cv2.erode(mask, kernel, iterations=2) 

# print(type(warped_image), type(mask))
# warped_image = cv2.bitwise_and(warped_image, mask)

# plt.imshow(warped_image)
# cv2.imwrite("warped2.png", warped_image)

