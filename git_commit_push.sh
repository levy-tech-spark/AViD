#!/bin/bash

# Check if commit message is provided as argument
if [ -z "$1" ]
  then
    echo "Please provide a commit message"
    exit 1
fi

# Perform git add
git add .

# Perform git commit with provided commit message
git commit -m "$1"

# Perform git push
git push
