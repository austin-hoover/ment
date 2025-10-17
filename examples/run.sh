#!/bin/bash

find . -type d -name "*.ipynb_checkpoints" -exec rm -rf {} \;
find . -type f -name "*.DS_Store" -exec rm -rf {} \;

for folder in ./*/; do
    cd $folder
    pwd
    # find . -type f -name "*.py"    -exec echo {} \; -exec python {} \;
    find . -type f -name "*.ipynb" -exec echo {} \; -exec jupyter nbconvert --inplace --ExecutePreprocessor.kernel_name=ment --execute {} \;
    cd ..
done
