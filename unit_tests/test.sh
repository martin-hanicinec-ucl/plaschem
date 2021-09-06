#!/usr/bin/env bash

echo "\nRunning the PlasChem unit tests:"
python -m unittest discover -s unit_tests/test_plaschem
echo "\nRunning the PyGmol unit tests:"
python -m unittest discover -s unit_tests/test_pygmol
echo "\nRunning the PyGMoFork unit tests:"
python -m unittest discover -s unit_tests/test_pygmofwork
echo "\nRunning the OptiChem unit tests:"
python -m unittest discover -s unit_tests/test_optichem
