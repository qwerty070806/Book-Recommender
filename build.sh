#!/usr/bin/env bash
# exit on error
set -o errexit

# 1. Install all the libraries
pip install -r requirements.txt

# 2. Run our script to create the database tables
python create_tables.py