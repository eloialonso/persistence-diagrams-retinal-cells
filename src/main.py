import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from clickable_image import ClickableImage
from utils import angle2intensity_in_ring, angular_smoothing, apply_mask, create_ring_mask, get_quantile, get_peaks, make_gif, persistence_diagram, plot_img, superimpose_ring, superimpose_ring_and_intersections, water_descent


def parse_arguments():
    parser = argparse.ArgumentParser(description="tda for corner detection in retinal cells images", formatter_class=argparse.RawTextHelpFormatter)

    # out folder
    parser.add_argument("--outf", default=Path("out"), type=Path, 
                        help="path to the experiment folder.")

    # source image
    parser.add_argument("--image", type=Path, default=Path("images/real_image.tif"),
                        help="path to the image to analyze")

    # Ring
    parser.add_argument("--radius", type=int, default=6,
                        help="Radius of the ring (default: 6). \nWARNING: if --typical_edge is specified, this argument is not used since we will heuristically choose the radius")
    parser.add_argument("--typical_edge", type=str, choices=["measure_an_edge", "count_cells"],
                        help="Method to estimate the 'typical' edge length. This value is used to heuristically choose the radius of the ring. If not specified, we take the value of --radius. \n\t- 'measure_an_edge': click on a edge of 'typical' length \n\t- 'count_cells': count the cells on the image and we deduce the typical edge length from it")
    parser.add_argument("--thickness", type=int,
                        help="Thickness of the ring (default: (2/3)*radius)")

    # Angular smoothing
    parser.add_argument("--size_smoothing", type=int, default=30,
                        help="Size of the sliding window for angular smoothing, in degree")
    parser.add_argument("--stride_smoothing", type=int, default=1,
                        help="Stride of the sliding window for angular smoothing, in degree")

    # Minimum 'lifetime' in the persistence diagram
    parser.add_argument("--threshold_min_lifetime", type=float, default=5,
                        help="min_lifetime (in persistence diagram) is heuristically computed as :\nmax(threshold, coef * range_intensity)\nwith range_intensity = max - min of the pixel values in the ring")
    parser.add_argument("--coef_min_lifetime", type=float, default=0.05,
                        help="min_lifetime (in persistence diagram) is heuristically computed as :\nmax(threshold, coef * range_intensity)\nwith range_intensity = max - min of the pixel values in the ring")

    # No gif
    parser.add_argument("--gif", action="store_true",
                        help="if specified, .gif file is generated.")

    return parser.parse_args()


def main():

    args = parse_arguments()
    
    args.outf.mkdir(exist_ok=True, parents=True)
    im = ClickableImage(args.image)

    # ******************* PARAMETERS ******************************************

    # Intensity of interest: intensity above which a pixel may be considered as a center / an edge
    # Assumption: centers and edges are brighter than background
    intensity_of_interest = get_quantile(im.img, 0.75)

    #
    # Ring
    #
    # - radius: if --typical_edge is not specified, we use the value of --radius, else we heuristically chose it from the 'typical edge length'
    # - thickness: if --thickness is not specified, we use (2/3)*radius
    if args.typical_edge == "measure_an_edge":
        typical_edge_length = im.measure_an_edge()
        print(f"You selected an edge of length : {typical_edge_length}")
        radius = math.ceil(typical_edge_length * 0.6)
    elif args.typical_edge == "count_cells":
        n_cells = im.count_cells()
        typical_edge_length = im.n_cells_to_edge_length(n_cells)
        radius = math.floor(typical_edge_length / 3)
    else:
        radius = args.radius
    thickness = args.thickness if args.thickness is not None else int((2 / 3) * radius)

    # *************************************************************

    #
    # Manual selection of the point to analyze
    #
    center = im.get_point()
    center = (math.ceil(center[0]), math.ceil(center[1]))
    print(f"Selected point: {center}")

    #************************** STEP 1 ****************************#
    # Filter the image with a ring centered on the point of interest
    #**************************************************************#
    #
    # define ring
    #
    ring = {}
    ring["center"] = center
    ring["radius"] = (radius - 0.5 * thickness, radius + 0.5 * thickness)

    #
    # filter image with the ring
    #
    mask = create_ring_mask(im.img, ring)
    img_filtered = apply_mask(im.img, mask)

    #
    # Visualization of the ring around the selected point
    #

    # image filtered by the ring
    fig, ax = plot_img(img_filtered, title="Image filtered by the ring")
    fig.savefig(args.outf / "1_img_filtered.png", bbox_inches="tight")

    # image with superimposed ring
    img_with_ring = superimpose_ring(im.img, ring, mask)
    fig, ax = plot_img(img_with_ring, title=f"Center : ({ring['center'][0]:.2f}, {ring['center'][1]:.2f}) \n Radius : ({ring['radius'][0]:.2f}, {ring['radius'][1]:.2f})")
    fig.savefig(args.outf / "2_ring.png", bbox_inches="tight")


    #********************** STEP 2 **********************#
    # Get the 'mountain' relief of intensity in the ring
    #****************************************************#

    angle2intensity = angle2intensity_in_ring(im.img, ring, mask)
    bucket2intensity, intensity2bucket = angular_smoothing(angle2intensity,
                                                           size=args.size_smoothing,
                                                           stride=args.stride_smoothing,
                                                           cast_to_int=False)

    #************************** STEP 3 *********************#
    # 'water descent' on our relief, to obtain the barcodes
    # of the peaks (birth/death)
    #
    # Let f be our 'relief' function, f: angle -> intensity
    # Compute persistence of the connected components of
    # the filtration {f^-1([h, +inf[), for all h real}
    #*******************************************************#
    barcodes = water_descent(intensity2bucket, args.stride_smoothing)

    # Heuristically computes a correct 'min_lifetime', a threshold on duration (death - birth):
    # - below: the connected components is considered as noise
    # - above: the cc is considered as a peak
    # this is the dashed line that "cuts" the persistence diagram
    range_intensity = max(bucket2intensity) - min(bucket2intensity)
    min_lifetime = max(args.threshold_min_lifetime, range_intensity * args.coef_min_lifetime)

    #****************************** STEP 4 ********************************#
    # For visualization, we report the barcodes in a 'persistence diagram'
    # We filter out the cc with a persistence < min_lifetime
    # The remaining ones are the significant 'peaks'
    # We count them to determine if the point of interest is:
    # - a corner (3 or more peaks)
    # - an edge (2 peaks)
    # - a part of background (else)
    #***********************************************************************#

    # Analysis of the barcodes: persistence diagram and type of the point
    peaks = get_peaks(barcodes, cut=min_lifetime)
    n_peaks = len(peaks)

    #
    # Visualization
    #
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 7.3))

    # intensity relief in the ring
    ax1.plot(range(0, 360, args.stride_smoothing), bucket2intensity)
    ax1.set_xlim(0, 360)
    ax1.set_xlabel("Angle in the ring (°)")
    ax1.set_ylabel("Intensity in [0, 255]")
    ax1.set_title("Intensity vs. Angle")

    # corresponding persistence diagram
    ax2 = persistence_diagram(ax2, barcodes, min_lifetime, intensity_of_interest)
    ax2.set_title(f"Persistence Diagram - {n_peaks} persistent CCs")
    ax2.set_xlabel("Birth intensity")
    ax2.set_ylabel("Death intensity")
    fig.savefig(args.outf / "3_persistence_diagram.png", dpi=90, bbox_inches='tight')

    #
    # classify the selected point
    #
    if n_peaks >= 3:
        point_type = "corner"
    elif n_peaks == 2:
        point_type = "edge"
    else:
        point_type = "background"
    print(f"\n==> Point type: {point_type} ({n_peaks} peak(s))")

    #
    # display detected edges
    #
    if n_peaks >= 2:
        angles_of_edges = []
        for cc in peaks:
            angle_of_edge = cc.peak.x
            angles_of_edges.append(angle_of_edge)
        img_with_ring_and_intersections = superimpose_ring_and_intersections(im.img, ring, angles_of_edges, mask)
        fig, ax = plot_img(img_with_ring_and_intersections, title=f"Angles between horizontal and detected edges\n{sorted(angles_of_edges)}")
        fig.savefig(args.outf / "4_edges.png")

    #
    # Make animated gif
    #
    if args.gif:
        make_gif(args.outf, barcodes, min_lifetime, bucket2intensity, args.stride_smoothing, intensity_of_interest)

    print(f"Visualizations stored in '{args.outf}'.")

    return


if __name__=="__main__":
    main()
