#!/bin/bash
echo
echo 'Creating additional folders .. '
echo 
mkdir models
mkdir log
mkdir plots
echo 'Downloading data (2.5GB, this might take a bit) .. '
echo
wget -N 'https://depositonce.tu-berlin.de/bitstreams/fe02c1e0-64d9-4a92-ac4d-a8a0ef455c8f/download'
echo 'Unpacking data .. '
echo
unzip download
rm download
