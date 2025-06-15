#!/bin/bash

# Optional: print Python version for debugging
python --version

# Install dependencies (if needed)
pip install -r requirements.txt

# Configure Git user details
git config --global user.name "niteowl1986"
git config --global user.email "49530755+niteowl1986@users.noreply.github.com"

# Run the setup script
python chat_newsbot_setup.py