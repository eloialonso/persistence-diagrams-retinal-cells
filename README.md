<h1 align="center">
  Topological Data Analysis for Corner Detection in images of Retinal Cells
</h1>

In images of retinal cells, we use persistence diagrams to classify a given pixel as a corner, an edge, etc.

<p align="center">
  <img alt="Persistence diagram" src="docs/persistence.gif">
</p>

## Principle

<p align="center">
  <img alt="Image of cells, chosen point and ring" src="docs/ring.png">
</p>

**Goal**: classify the selected pixel (red) as belonging to a *corner*, an *edge* or simply *background*. 

**How**: we consider the intensity of the pixels (blue) located in a ring centered on our point of interest. Then, use the following set of rules: 
- if the intensity in the ring has two main peaks, then our point belongs to an edge
- if the intensity in the ring has three or more major peaks, our point belongs to a corner
- else, our point belongs to background.

**Difficulty**: define the idea of a *major* peak of intensity (versus a *noisy* peak).

**Approach**: use *persistence* to quantitatively differentiate between noisy peaks and major peaks.   

## Persistence Diagram

In the ring, consider the mapping *angle -> intensity* as a mountain relief, where the angle is the horizontal position and the intensity is the altitude. Fill this landscape with water until the highest mountain is covered up. Then, slowly empty the water and keep track of two events:

- the emergence of a mountain, like an island = the **birth** of the mountain
- when the water level is low enough, the merging of two previously-separated islands = the **death** of the smaller mountain

Then, represent each mountain in the persistence diagram as a point [birth, death]. During the water descent, *noisy* peaks will emerge and quickly get merged with a higher peak: they have a small persistence. On the contrary, *major* peaks are characterized by a longer persistence. Therefore, in the persistence diagram, we found the noisy peaks close to the diagonal, and the major peaks farer from the diagonal. 

<p align="center">
  <img alt="Intensity in the ring and corresponding persistence diagram" src="docs/persistence_diagram.png">
</p>

*Left*: the intensity of the pixels in the ring as a function of their angle in the ring. *Right*: the corresponding persistence diagram. **In this example, we count 4 major peaks and deduce that the point of interest is located on a corner**. 

A bit more formally, let *f* be the function plotted on the left (mapping angle to intensity). For a given intensity *x* in *[0, 255]*, we consider the **connected components** of the following set:

<img src="docs/eq_preimage.svg" alt=""/>

Then, we vary the intensity *x* from 255 down to 0, tracking the connected components at each step, recording their birth and their death. The persistence diagram is the graphical representation of a connected component's life: each point corresponds to a connected component, with coordinates [intensity of birth, intensity of death].

## Getting started

### Installation

- Clone the project
```bash
git clone https://github.com/bloodymosquito/tda-retinal-cells
```

- Create a virtual environment (optional)
```bash
virtualenv -p python3 <location_of_the_new_virtualenv>
source <location_of_the_new_virtualenv>/bin/activate
```

- Install the necessary packages (numpy, matplotlib, opencv)
```bash
cd tda-retinal-cells
pip install -r requirements.txt
```

### Basic use

```bash
python main.py
```

This will display a default image of cells. Select a point to run the analysis on. The output images are stored in `./out/`. 

### Options

It is possible to change the default image, the default values of the hyper parameters (shape of the ring, minimum persistence to separate noisy peaks vs. major peaks etc.). Enter `python main.py -h` for details.

