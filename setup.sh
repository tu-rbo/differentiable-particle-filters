#!/bin/bash
echo
echo 'Creating additional folders .. '
echo 
mkdir models
mkdir log
mkdir plots
echo 'Downloading data (2.5GB, this might take a bit) .. '
echo
wget -N 'https://tubcloud.tu-berlin.de/s/0fU32cq0ppqdGXe/download'
echo 'Unpacking data .. '
echo
unzip download
rm download
