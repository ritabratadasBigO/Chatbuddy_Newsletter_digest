#!/bin/bash

# Optional: print Python version for debugging
python --version

# Install dependencies (if needed)
pip install -r requirements.txt

# Configure Git user details
git config --global user.name "ritabratadasBigO"
git config --global user.email "49530755+ritabratadasBigO@users.noreply.github.com"

# Ensure origin is configured
git remote get-url origin 2> /dev/null
if [ $? -ne 0 ]; then
  git remote add origin https://${GITHUB_PAT}@github.com/ritabratadasBigO/Chatbuddy_Newsletter_digest.git
fi

# Ensure we are on the main branch to avoid detached HEAD issues
git checkout main
git pull origin main

# Run the setup script
python chat_newsbot_setup.py