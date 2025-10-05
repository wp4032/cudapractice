#!/bin/bash

# Check if Nsight Systems is already installed
if command -v nsys &> /dev/null; then
    echo "Nsight Systems is already installed"
    nsys --version
    exit 0
fi

echo "Nsight Systems not found. Installing..."

# Download Nsight Systems
DEB_FILE="NsightSystems-linux-cli-public-2025.5.1.121-3638078.deb"
if [ ! -f "$DEB_FILE" ]; then
    echo "Downloading Nsight Systems..."
    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_5/NsightSystems-linux-cli-public-2025.5.1.121-3638078.deb
fi

# Install the package
echo "Installing Nsight Systems..."
sudo dpkg -i "$DEB_FILE"

# Fix any dependency issues
sudo apt-get install -f -y

echo "Installation complete!"
nsys --version
