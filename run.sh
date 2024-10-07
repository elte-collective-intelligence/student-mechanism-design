#!/bin/bash

# Navigate to the script's directory (optional)
cd "$(dirname "$0")"

# Run the container with bind mount
apptainer run --bind ./:/app scyard.sif
