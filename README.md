# Source code release

This repository contains the source code and scripts used to generate the results presented in the paper 

Honório, H.T., Franceschini, A., Ferronato, M., Hajibeygi, H. *Salt cavern simulations with a stabilized mixed finite element formulation for low-order tetrahedral elements*, CMAME, 2026.

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
