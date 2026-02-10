# Salt cavern simulations with a stabilized mixed finite element formulation for low-order tetrahedral elements

This repository contains the source code accompanying the paper:

> **Salt cavern simulations with a stabilized mixed finite element formulation for low-order tetrahedral elements**  
> Hermínio T. Honório, Andrea Franceschini, Massimiliano Ferronato, and Hadi Hajibeygi  
> *(submitted to Computer Methods in Applied Mechanics and Engineering)*  

The purpose of this repository is to ensure **reproducibility**, **transparency**, and **reusability** of the numerical methods and results presented in the paper.

---

## Scope of the repository

This repository provides:

- Implementation of the **stabilized mixed finite element formulation**
- Reproducible **numerical examples** used in the paper
- Scripts for **mesh handling**, **problem setup**, and **post-processing**
- Reference input files and configuration parameters

The code is intended for **research and educational purposes** and reflects the formulations and assumptions described in the paper.

---

## Requirements

This project requires **FEniCSx 0.9.0**, which cannot be installed via pip and it can only be installed on Ubuntu. If you use Windows, you must install [WSL](https://learn.microsoft.com/en-us/windows/wsl/) first.

Install FEniCSx using one of the following methods:

- Conda (recommended):
  ```bash
  conda install -c conda-forge fenics-dolfinx=0.9.0 mpich pyvista
- Docker
  ```bash
  docker run -it ghcr.io/fenics/dolfinx/dolfinx:v0.9.0


After installing FEniCSx 0.9.0, install the Python dependencies:

```bash

pip install -r requirements.txt
```




