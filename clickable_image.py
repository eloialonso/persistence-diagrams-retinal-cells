#! /usr/bin/env python
# coding: utf-8

"""
We define here ClickableImage, that stores an image and allows some clicking/selecting operations on it (for user interaction).  
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


class ClickableImage:

    def __init__(self, fname):
        r"""Clickable image, created from an image file.
        Arguments:
            fname (str): path to an image       
        """
      
        self.fname = fname

        # load image and convert it from BGR to GRAY
        self.img = cv2.cvtColor(cv2.imread(self.fname), cv2.COLOR_BGR2GRAY)

        self.point = ()
        self.points = []

    def __repr__(self):
        return "MyImage : name = {}, shape = {}".format(self.fname.split("/")[-1], self.img.shape)

    def __onclick__(self, click):
        r"""Register the mouse position on click event"""
        self.point = (click.xdata, click.ydata)
        self.points.append(self.point)
        return self.point

    def get_point(self, n_point=1, title="Click on a point"):
        r"""The user select a point with his mouse"""
        self.points = []
        fig = plt.figure()
        fig.add_subplot(111)
        fig.suptitle(title)
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB))
        fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        for _ in range(n_point):
            plt.waitforbuttonpress()
        plt.close()
        return (self.point[1], self.point[0])

    def measure_an_edge(self):
        r"""The user selects two points on image to obtain a typical edge length"""
        self.get_point(n_point=2, title="Select an edge by clicking on 2 points")
        length = int(np.sqrt((self.points[1][0] - self.points[0][0])**2 + (self.points[1][1] - self.points[0][1])**2))
        return length

    def count_cells(self):
        r"""Display the image for the user to (approximately) count the number of cells"""
        self.get_point(title="Count the cells, then click on image")
        while True:
            n_cells = input("How many cells did you see on this image ? ")
            try:
                n_cells = int(n_cells)
            except ValueError :
                if n_cells == "exit":
                    return
                print("You must enter a valid number")
                continue
            if n_cells <= 0:
                print("You must enter a positive number")
                continue
            break
        return n_cells

    def n_cells_to_edge_length(self, n_cells):
        r""" Heuristic: deduce the typical edge length from the number of cells,
        with the (strong) assumption that all cells are perfect hexagons.
        Arguments:
            n_cells: approximate number of cells on the image
        """
        typical_edge_length = int(np.sqrt(2 * np.product(self.img.shape) / (3 * np.sqrt(3) * n_cells)))
        return typical_edge_length






