#! /usr/bin/env python
# coding: utf-8


r"""
Diverse useful functions: ring-filtering of the image, water descent, persistence diagram etc.
"""


import math

import cv2
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from connected_component import ConnectedComponent
from point import Point


def get_quantile(image, perc):
    r"""Return the chosen quantile of the image
    Arguments:
        image: np.array
        perc: in [0, 1]
    """
    assert perc >= 0 and perc < 1, "perc must be in [0, 1]"
    return sorted(image.reshape(-1))[int(perc*np.product(image.shape))]


def create_ring_mask(img, ring):
    r"""Prepare the mask to filter an image with a ring at a given point
    Arguments:
        img: 2D np array
        ring: dict containing 'center': (x, y) and 'radius': (r1, r2)
    """
    mask = np.ones(shape=img.shape).astype(bool)
    center = ring["center"]
    y, x = np.ogrid[-center[0]: img.shape[0] - center[0], -center[1]: img.shape[1] - center[1]]

    # mask is 1 on the ring, 0 elsewhere
    mask = (x * x + y * y <= np.max(ring["radius"])**2) & (x * x + y * y >= np.min(ring["radius"])**2)
    return mask


def apply_mask(img, mask):
    r"""Filter an image with a ring at a given point
    Arguments:
        img: 2D np array
        ring: dict containing 'center': (x, y) and 'radius': (r1, r2)
    """
    out = np.copy(img)
    out[np.logical_not(mask)] = 0
    return out


def plot_img(img, title=""):
    r"""Plot an image, and optionally add a title"""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        assert len(img.shape) == 3 and img.shape[-1] == 3, "invalid image format"
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.imshow(img)
    return fig, ax


def simple_plot(x, y, title=""):
    r"""Simple plot y versus x"""
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.plot(x, y)
    return fig, ax


def angle2intensity_in_ring(img, ring, mask=None):
    r"""Matches each degree in the ring to the intensities of the pixels in this sector
    Arguments:
        img: 2D np array
        ring: dict containing 'center': (x, y) and 'radius': (r1, r2)
        mask: mask of ring filter
    """
    center = ring["center"]

    if mask is None:
        mask = create_ring_mask(img, ring)

    img_filtered = apply_mask(img, mask)

    # discretisation of the ring
    angle2intensity = {}
    for i, j in zip(np.nonzero(mask)[0], np.nonzero(mask)[1]):
        if i != center[0]:    # when x != 0
            angle = int((180/np.pi)*np.arctan((j-center[1])/(i-center[0]))) % 360   # angular resolution of 1 degree (int());  conversion radian to degree (180/pi then %360) ;  calculation of the angle (arctan)
            if i < center[0] :  # because of arctan, 3 cases to be considered : when x < 0 [...]
                if j > center[1]:
                    angle = angle - 180
                if j <= center[1]:
                    angle = angle + 180
        elif j > center[1]:   #[...] when x = 0 and y > 0
            angle = 90
        else:    #[...] when x = 0 and y < 0
            angle = 270
        angle2intensity.setdefault(angle, []).append(img_filtered[i, j])
    return angle2intensity


def angular_smoothing(angle2intensity, size, stride, cast_to_int=True):
    r"""Angular smoothing of the relation angle --> list of intensity, with a sliding window
    Arguments:
        angle2intensity: dict, output of angle2intensity_in_ring
        size (int): size of the sliding window, in degree
        stride (int): stride of the sliding window
        cast_to_int (bool): if True, cast to int
    """
    bucket2intensity = np.zeros(math.ceil(360 / stride))
    intensity2bucket = {}
    for i, bucket_number in enumerate(range(0, 360, stride)) : #bucket_number = i * stride
        count = 0
        for j in range(bucket_number - size // 2, bucket_number + size // 2):
            if j >= 360:
                j -= 360
            if j < 0:
                j += 360
            if j in angle2intensity.keys():
                for intensity in angle2intensity[j]:
                    count += 1
                    bucket2intensity[i] = bucket2intensity[i] + intensity # point[0] corresponds to the intensity at angle j
        if count != 0:
            bucket2intensity[i] = (1/count) * bucket2intensity[i]
            if cast_to_int:
                bucket2intensity[i] = int(bucket2intensity[i])
            intensity2bucket.setdefault(bucket2intensity[i], []).append(bucket_number)  #key:intensity  value:list of angles which all have intensity(key) as intensity(quantity)

    return bucket2intensity, intensity2bucket


def superimpose_ring(img, ring, mask=None):
    r"""Superimpose the selected point (red) and the ring around it (blue).
    Arguments:
        img: 2D np array
        ring: dict containing 'center': (x, y) and 'radius': (r1, r2)
        mask: mask of ring filter
    """
    if mask is None:
        mask = create_ring_mask(img, ring)
    center = ring["center"]
    im_with_ring = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    im_with_ring[center[0], center[1]] = np.array([200, 0, 0])  # center
    im_with_ring[mask] = np.array([0, 200, 200])                # ring area
    return im_with_ring


def compute_angle(center, point):
    r"""Compute the angle between horizontal and line passing by 'center' and 'point'"""
    x, y = center
    i, j = point
    if i != x:
        # compute the angular position of the pixel within the ring
        angle = int((180 / np.pi) * np.arctan((j - y) / (i - x))) % 360
        if i < x :
            if j > y:
                angle = angle - 180
            else:
                angle = angle + 180
    elif j > y:
        angle = 90
    else:
        angle = 270
    return angle


def superimpose_ring_and_intersections(img, ring, angles, mask=None, alpha=8):
    r"""Superimpose the selected point (red), the ring around it (blue) and the regions of intersection (red) with edges.
    Arguments:
        img: 2D np array
        ring: dict containing 'center': (x, y) and 'radius': (r1, r2)
        angles: list of angles of edges
        mask: mask of ring filter
        alpha: value in degree of the angle of the half cone added to angles
    """
    if mask is None:
        mask = create_ring_mask(img, ring)

    # For visibility purpose, consider angles and angles +- alpha
    angles_extended = list(angles)
    for angle in angles:
        angles_extended.extend(range(angle - alpha, angle + alpha + 1))

    # Identify pixels of the ring that are on an intersection
    pixels_on_intersection = np.zeros_like(img)
    for pixel in zip(np.nonzero(mask)[0], np.nonzero(mask)[1]):
        angle = compute_angle(ring["center"], pixel)
        if angle in angles_extended:
            pixels_on_intersection[pixel[0], pixel[1]] = 1

    # Superimpose the ring in one color, and the regions of intersection in another color
    im_with_ring = superimpose_ring(img, ring, mask)
    im_with_ring[pixels_on_intersection.astype(bool)] = np.array([200, 0, 0])

    return im_with_ring


def water_descent(intensity2bucket, stride):
    r"""Performs the 'water descent' and stores the info/barcode of the peaks (birth/death)
    Arguments:
        intensity2bucket (dict)
        stride (int)
    """
    ConnectedComponent.reset()

    # Start from maximum intensity then decrease
    intensities = sorted(list(intensity2bucket.keys()))[::-1]
    for intensity in intensities:

        # Get all points for this intensity
        angles = intensity2bucket[intensity]
        for angle in angles:
            P = Point(x=angle, y=intensity, cc=[])

            # Check if the point (angle, intensity) is close to the left / right border of the relief
            P_close_to_right = False
            P_close_to_left = False
            if P.x - stride < min(min(intensity2bucket.values())):  #P is close to left border
                P_close_to_left = True
            if P.x + stride > max(max(intensity2bucket.values())):  #P is close to right border
                P_close_to_right = True

            # If P is close to an existing cc, add P to the cc
            for cc in ConnectedComponent.connected_components:
                is_close, on_left, on_right = cc.is_close_to_point(P, stride)
                if is_close:
                    P.belongs_to.append(cc)
                    cc.add_member(P, on_left=on_left, on_right=on_right)
                if P_close_to_left :
                    P_shift = Point(x=P.x + max(max(intensity2bucket.values())) - min(min(intensity2bucket.values())), y=P.y, cc=[])
                if P_close_to_right:
                    P_shift = Point(x=P.x - max(max(intensity2bucket.values())) + min(min(intensity2bucket.values())), y=P.y, cc=[])
                if P_close_to_left or P_close_to_right:
                    if cc.is_inversed:
                        continue
                    else:
                        is_close, on_left, on_right = cc.is_close_to_point(P_shift, stride)
                        assert not (on_left or on_right)
                        if is_close:
                            P.belongs_to.append(cc)
                            cc.add_member(P, point_close_left=P_close_to_left, point_close_right=P_close_to_right)
            P.belongs_to = list(set(P.belongs_to))

            # if P does not belong to any cc, create a new cc whose peak (and only member) is P
            if P.belongs_to == []:
                cc = ConnectedComponent(peak=P, x_left=P.x, x_right=P.x, members=[P])
                P.belongs_to.append(cc)

            # if P is close to different cc, merge the different connected components
            elif len(P.belongs_to) >= 2:
                cc = P.belongs_to[0]
                for i in range(1,len(P.belongs_to)):
                    cc = ConnectedComponent.union(cc, P.belongs_to[i], P.y, stride=stride)
                P.belongs_to = [cc]

    # At the end of the water descent, if there is more than one cc remaining, merge them together
    final_cc = ConnectedComponent.connected_components.copy()
    cc = final_cc[0]
    for i in range(1, len(final_cc)):
        try:
            print("Warning : connected component with peak ({}, {}) wasn't merge at the end".format(final_cc[i].peak.x, final_cc[i].peak.y))
            cc = ConnectedComponent.union(cc, final_cc[i], intensity, stride=stride)
        except IndexError:
            print(i)
    ConnectedComponent.history[ConnectedComponent.connected_components[0]].append(intensity)

    return ConnectedComponent.history


def compute_dist_to_diag(barcodes):
    r"""Compute distance of each cc (birth, death) to the diagonal (birth = death)"""
    history = np.array(list(barcodes.values()))
    distances = (history[:, 0] - history[:, 1]) / np.sqrt(2)
    distances = np.around(distances, decimals=1)
    dist_to_diag = {}
    for i, cc in enumerate(list(barcodes.keys())):
        dist_to_diag[cc] = distances[i]
    return dist_to_diag


def get_peaks(barcodes, cut):
    r"""Select the barcodes with lifetime > cut"""
    peaks = []
    dist_to_diag = compute_dist_to_diag(barcodes)
    peaks = [cc for cc in barcodes if dist_to_diag[cc] > cut]
    return peaks


def persistence_diagram(ax, barcodes, cut, intensity_of_interest=0):
    r"""Visualization of barcodes with a persistence diagram"""
    dist_to_diag = compute_dist_to_diag(barcodes)
    
    peaks = np.array([v for cc, v in barcodes.items() if dist_to_diag[cc] > cut])
    noise = np.array([v for cc, v in barcodes.items() if dist_to_diag[cc] <= cut])

    y_min = np.min(list(barcodes.values()))
    y_max = np.max(list(barcodes.values()))
    if y_max < intensity_of_interest :
        y_max = intensity_of_interest

    # draw the persistence diagram
    ax.set_aspect('equal')
    ax.set_xlim([y_min, y_max])
    ax.set_ylim([y_min, y_max])
    ax.plot([y_min + np.sqrt(2) * cut, y_max], [y_min, y_max - np.sqrt(2) * cut], 'r--')
    ax.add_patch(patches.Rectangle((y_min, y_min), y_max - y_min, y_max - y_min, alpha=0.1))
    ax.fill_between(x=np.arange(y_max, y_min, -0.1), y1=y_max, y2=np.arange(y_max, y_min, -0.1))
    if noise.size > 0:
        ax.scatter(noise[:, 0], noise[:, 1], c='black', s=50)
    if peaks.size > 0:
        ax.scatter(peaks[:, 0], peaks[:, 1], c='brown', s=150)
    return ax


def make_gif(outf, barcodes, cut, bucket2intensity, stride, intensity_of_interest=0):
    r"""Make persistence diagram gif"""

    dist_to_diag = compute_dist_to_diag(barcodes)

    ccs = list(barcodes.keys())
    all_ = np.array(list(barcodes.values()))

    idxs = all_[:, 1].argsort()[::-1] # sort by decreasing level of birth
    all_ = all_[idxs] 
    ccs = list(barcodes.keys())
    ccs = [ccs[i] for i in idxs]
    is_peak = np.array([dist_to_diag[cc] > cut for cc in ccs])
    
    y_min = np.min(list(barcodes.values()))
    y_max = np.max(list(barcodes.values()))
    if y_max < intensity_of_interest :
        y_max = intensity_of_interest

    angles = np.array(range(0, 360, stride))

    # draw the successive persistence diagram
    with imageio.get_writer(outf / "persistence_diagram.gif", mode='I', duration=0.05) as writer:
        for i in tqdm(range(len(all_)), desc="Generating GIF"):
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 7.3))

            current_intensity = all_[i, 1]

            # intensity relief in the ring
            ax1.plot(angles, bucket2intensity, color='black')
            ax1.set_xlim(0, 360)
            ax1.add_patch(patches.Rectangle((0, y_min), 360, current_intensity - y_min, alpha=0.7))
            ax1.set_xlabel("Angle in the ring (°)")
            ax1.set_ylabel("Intensity in [0, 255]")

            # persistence diagram
            ax2.set_aspect('equal')
            ax2.set_xlim([y_min, y_max])
            ax2.set_ylim([y_min, y_max])
            ax2.plot([y_min + np.sqrt(2) * cut, y_max], [y_min, y_max - np.sqrt(2) * cut], 'r--')
            ax2.add_patch(patches.Rectangle((y_min, y_min), y_max - y_min, y_max - y_min, alpha=0.1))
            ax2.fill_between(x=np.arange(y_max, y_min, -0.1), y1=y_max, y2=np.arange(y_max, y_min, -0.1))
            
            peaks = all_[:i+1][is_peak[:i+1]]
            noise = all_[:i+1][np.logical_not(is_peak[:i+1])]

            if len(peaks) > 0:
                ax2.scatter(peaks[:, 0], peaks[:, 1], c='brown', s=150)
            if len(noise) > 0:
                ax2.scatter(noise[:, 0], noise[:, 1], c='black', s=50)
            
            ax2.set_xlabel("Birth intensity")
            ax2.set_ylabel("Death intensity")

            if i == len(all_) - 1:
                for p in peaks:
                    circle = plt.Circle((p[0], p[1]), 5, color='b', fill=False)
                    ax2.add_patch(circle)

            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(img)
            plt.close()

        for _ in range(60):
            writer.append_data(img)

    return

