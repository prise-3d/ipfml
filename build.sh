#! bin/bash

# script used to build documentation from setup.py build command
echo "Build package..."
python setup.py build

echo "Build documentation..."
rm -r docs/source/ipfml
cd docs && make clean && make html
