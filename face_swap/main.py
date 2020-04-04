#! /usr/bin/env python
import os
import cv2
import argparse

import numpy as np
import glob
from face_detection import face_detection
from face_points_detection import face_points_detection
from face_swap import warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colours, transformation_from_points
import utils
from scipy.spatial import Delaunay

root_dir = "./output_images/"
face_dir = root_dir + "face/"
left_eye_dir = root_dir + "left_eye/"
right_eye_dir = root_dir + "right_eye/"
mouth_dir = root_dir + "mouth/"
fragment_dir = root_dir + "/triangles/triangle_%012d/"
bleed_factor = 5
def select_face(im, r=10):
    faces = face_detection(im)

    if len(faces) == 0:
        print('Detect 0 Face !!!')
        exit(-1)

    if len(faces) == 1:
        bbox = faces[0]
    else:
        bbox = []

        def click_on_face(event, x, y, flags, params):
            if event != cv2.EVENT_LBUTTONDOWN:
                return

            for face in faces:
                if face.left() < x < face.right() and face.top() < y < face.bottom():
                    bbox.append(face)
                    break
        
        im_copy = im.copy()
        for face in faces:
            # draw the face bounding box
            cv2.rectangle(im_copy, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 1)
        cv2.imshow('Click the Face:', im_copy)
        cv2.setMouseCallback('Click the Face:', click_on_face)
        while len(bbox) == 0:
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        bbox = bbox[0]

    points = np.asarray(face_points_detection(im, bbox))
    
    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)
    
    x, y = max(0, left-r), max(0, top-r)
    w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]


def main(src_image_filename):
    print(src_image_filename, flush=True)

    # Read images
    src_img = cv2.imread(src_image_filename)
    dst_img = cv2.imread('face.jpg')

    # Select src face
    src_points, src_shape, src_face = select_face(src_img)
    # Select dst face
    dst_points, dst_shape, dst_face = select_face(dst_img)

    h, w = dst_face.shape[:2]
    
    ### Warp Image
    if True:
        ## 3d warp
        warped_src_face = warp_image_3d(src_face, src_points[:48], dst_points[:48], (h, w))
    else:
        ## 2d warp
        src_mask = mask_from_points(src_face.shape[:2], src_points)
        src_face = apply_mask(src_face, src_mask)
        # Correct Color for 2d warp
        warped_dst_img = warp_image_3d(dst_face, dst_points[:48], src_points[:48], src_face.shape[:2])
        src_face = correct_colours(warped_dst_img, src_face, src_points)
        # Warp
        warped_src_face = warp_image_2d(src_face, transformation_from_points(dst_points, src_points), (h, w, 3))

    ## Mask for blending
    mask = mask_from_points((h, w), dst_points)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask*mask_src, dtype=np.uint8)

    ## Correct color
    warped_src_face = apply_mask(warped_src_face, mask)
    dst_face_masked = apply_mask(dst_face, mask)
    warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)
    
    ## Shrink the mask
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    ##Poisson Blending
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

    left_eye_mask = mask_from_points((h, w), dst_points[36:42])
    right_eye_mask = mask_from_points((h, w), dst_points[42:48])
    mouth_mask = mask_from_points((h, w), dst_points[48:60])
    face_mask = mask_from_points((h, w), dst_points[:27])

    triangle_masks = []
    for indexes in Delaunay(dst_points).simplices:
        ignore = False
        for ignore_points in [
            range(36, 41),
            range(42, 47),
            range(48, 59),
            range(0,26)
        ]:
            if all(e in ignore_points for e in indexes):
                ignore = True

        if not ignore:
            mask = mask_from_points((h, w), dst_points[indexes])
            triangle_masks.append(mask)

    x, y, w, h = dst_shape

    white_image = np.zeros(warped_src_face.shape, np.uint8)
    white_image[:,:,:] = (255,255,255)

    dst_img_cp = dst_img.copy()
    #dst_img_cp[y:y+h, x:x+w] = output
    #white_image[y:y+h, x:x+w] = warped_src_face
    #output = dst_img_cp
    #output = warped_src_face #white_image
    kernel = np.ones((3,3), np.uint8) 

    ## face
    diluted_mask = apply_mask(white_image, face_mask)[:,:,1]
    output = apply_mask(warped_src_face, diluted_mask)
    channels = list(cv2.split(output)) + [diluted_mask]
    output = cv2.merge(channels)
    utils.save_image(face_dir, output)

    ## left eye
    output = apply_mask(white_image, left_eye_mask)
    diluted_mask = cv2.dilate(output, kernel, iterations=bleed_factor)[:,:,1] 
    output = apply_mask(warped_src_face, diluted_mask)
    channels = list(cv2.split(output)) + [diluted_mask]
    output = cv2.merge(channels)
    utils.save_image(left_eye_dir, output)

    ## left eye
    output = apply_mask(white_image, right_eye_mask)
    diluted_mask = cv2.dilate(output, kernel, iterations=bleed_factor)[:,:,1] 
    output = apply_mask(warped_src_face, diluted_mask)
    channels = list(cv2.split(output)) + [diluted_mask]
    output = cv2.merge(channels)
    utils.save_image(right_eye_dir, output)

    ## mouth
    output = apply_mask(white_image, mouth_mask)
    diluted_mask = cv2.dilate(output, kernel, iterations=bleed_factor)[:,:,1] 
    output = apply_mask(warped_src_face, diluted_mask)
    channels = list(cv2.split(output)) + [diluted_mask]
    output = cv2.merge(channels)
    utils.save_image(mouth_dir, output)

    for index, triangle_mask in enumerate(triangle_masks):
        diluted_mask = apply_mask(white_image, triangle_mask)[:,:,1] 
        output = apply_mask(warped_src_face, diluted_mask)
        channels = list(cv2.split(output)) + [diluted_mask]
        output = cv2.merge(channels)
        folder = fragment_dir % index
        utils.save_image(folder, output)

    #transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)

    # dir_path = os.path.dirname(args.out)
    # if not os.path.isdir(dir_path):
    #     os.makedirs(dir_path)

    #cv2.imwrite(args.out, output)

    # ##For debug
    # if not args.no_debug_window:
    #     cv2.imshow("From", dst_img)
    #     cv2.imshow("To", output)
    #     cv2.waitKey(0)
        
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    latest_index = utils.get_latest_index(face_dir)
    for src_image_filename in list(glob.glob(f"./input_images/*.png"))[latest_index:]:
        main(src_image_filename)
