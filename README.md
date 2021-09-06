## Introduction:

This is a repository accompanying part I. of my PhD thesis:

*Hanicinec, M. 2021, Towards Automatic Generation of Chemistry Sets for Plasma Modeling Applications, University College London, United Kingdom.*

The repository contains python packages implementing plasma global modeling and chemistry reduction framework.

## Installation:
The code in this repository is written in Python, and `pipenv` is used to handle the virtual environment management.
Although the `setup.py` file is present and the packages should be installable by pip straight from github, this has not been tested, and manual instalation of the dependencies would be required.
The following is a tested way to use the code on Linux operating systems:

* After cloning or downloading the repository, create the virtual environment from the `Pipfile.lock` containing all the necessary python packages by running 
  ```
  pipenv sync
  ```
  from the directory of the project, where the `Pipfile.lock` is located.
  This should result in the *completely identical* environment like the one used for the model development.
  The environment requires Python 3.9, which needs to be installed already.
* Once created, the environment can be activated by running 
  ```
  pipenv shell
  ```
  from its root directory (containing `Pipenv` file).
* Now the packages might be installed into the virtual environment path by running (in the virtual environment shell)
  ```
  pip install -e .
  ```
  from the repository root (with `setup.py` file).
  This installs the package in editable mode. 
  Any changes made to the code will immediately apply across the system.
  It also means the repository folder cannot be deletedor moved without breaking the install.
* Verify the installation of the packages succeeded by running the unit tests and run tests (by executing the prepared shell scripts):
  ```
  sh unit_tests/test.sh
  sh run_tests/test.sh
  ```

## Packages description:
This README file and the description is not a package documentation. 
I have tried my best to document directly in the code.
The python packages published in this repository served me personaly to acomplish my PhD goals, but were never intended to be released as stand-alone packages into the open-source community, with full documentation and support.
The following is a brief overview of the packages and their place in the global modeling and chemistry reduction framework.
* `plaschem` (__plas__ma __chem__istry):
  A package implementing data-structures for representation of species, reactions, and chemistry sets.
  Contains some handy functionality, such as the ability to self-consistently remove reactions or species from the chemistry, pretty-printing the chemistry sets, storing the sets into xml files, export chemistry sets as LaTeX tables, etc.
  Instances of the `plaschem.chemistry.Chemistry` class are used as one of the inputs for classes of the `pygmo_fwork` package.
* `pygmo_fwork` (__py__thon __g__lobal __mo__deling __f__rame__work__) package consists of two sub-packages:
  * `pygmol` (__py__thon __g__lobal __mo__de__l__) - a global model python package. The model uses `scipy.solve_ivp` solver to solve the set of ODEs defined in the `equations` module.
    The main module to control the global model is `model` and its class `Model`.
    `model.Model` class handles only the solver outputing raw data, no logging or post-processing functionality is present in the package.
  * `optichem` (__opti__mization of __chem__istries) - a package implementing the chemistry graph-based species ranking, the iterative skeletal reduction method, and Morris method of sensitivity analysis.
  Apart from the two sub-packages, `pygmo_fwork` package also implements a higher-level wrapper around the `pygmol` sub-package, which handles results logging and results post-processing.
  This is located in the `pygmo_fwork.global_model` module.
  The `pygmo_fwork.global_model.GlobalModel` class was written with interchangable model backends in mind, but as it stands, the `pygmol` model is the only backend available.
  The `GlobalModel` instances are used as inputs for the `optichem` classes, when running reduction or sensitivity analysis.

The source code itself contains quite extensive documentation, and I will be happy to supply additional documentation or example usage upon request.

