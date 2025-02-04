#!/bin/bash

set -x

for filename in *.ipynb; do
  if [[ ! "$filename" == *".ipynb_checkpoints"* ]]; then
    jupyter nbconvert --inplace --ExecutePreprocessor.kernel_name=ment --execute $filename;
  fi
done

for filename in *.py; do
  if [[ ! "$filename" == *".ipynb_checkpoints"* ]]; then
    python $filename;
  fi
done