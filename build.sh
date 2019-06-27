#! bin/bash

# Format code
echo "Use of yapf package to format code.."
yapf -ir -vv ipfml

# Build IPFML package
echo "Build package..."
python setup.py build

echo "Build documentation..."
rm -r docs/source/ipfml
cd docs && make clean && make html