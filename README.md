# xchimes

###  Repository for the paper "A Generalized Machine-Learning Framework for Developing Alchemical Many-Body Interaction Models for Polymer Grafted Nanoparticles"

We present the source code to perform Forward-Reverse (FR) method for potential of mean force (PMF) generation using HOOMD-Blue and to use ChIMES-Calulater in HOOMD-Blue to conduct coarse-grained (CG) molecular dynamics simulation.

# Prerequisites

The source code requires the following packages:

* [HOOMD-Blue](https://github.com/glotzerlab/hoomd-blue)
* [ChIMES-Calculator](https://github.com/rk-lindsey/chimes_calculator)
* [freud](https://github.com/glotzerlab/freud)
* [gsd](https://github.com/glotzerlab/gsd)
* [Numpy](https://github.com/numpy/numpy)
* [matplotlib](https://github.com/matplotlib/matplotlib)

# Usage
We explain the usage of three folders:

1.`fr_method`:

The Jupyter notebooks, `fr_demo_2b.ipynb` and `fr_demo_3b.ipynb` demonstrate the example and procedure to calculate two- and three-body PMF using the principal of FR method, where we use the Lennard-Jones particle as example for simpilicity. The steered molecular dynamics (SMD) method is used to calculate the work along reaction coordinate.

After running the two notebooks, it will generate two output for each notebook, including the SMD results: `smd.txt` and `smd_3b.txt`, as well as the SMD trajectories: `traj.gds` and `traj_3b.gsd`.

To calculate the three-body PMF, an interpolation method must be used to obtain the surrogate model for the two-body PMF and conduct the three-body PMF calculation. Here, we use the ChIMES model as interpolation method.
# Cite this work