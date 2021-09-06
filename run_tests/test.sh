#!/usr/bin/env bash

echo "\nRunning the PyGMol run tests:"
python -m unittest discover -s run_tests/test_pygmol --verbose
echo "\nRunning the PyGMo_Fwork run tests:"
python -m unittest discover -s run_tests/test_pygmofwork --verbose
