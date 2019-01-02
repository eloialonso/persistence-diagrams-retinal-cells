<h1 align="center">
  Topological Data Analysis for corner detection
</h1>

Persistence diagrams to classify pixels as corners, edges, etc. in images of retinal cells.

## Principle



<p align="center">
  <img alt="Persistence diagram" src="persistence.gif">
</p>


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

This will display a default image of cells. Select a point to run the analysis on. 

### Options

