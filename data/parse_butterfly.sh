#!/bin/bash

unzip *.zip -d butterfly
source ./bin/activate
python process_butterfly_videos.py
deactivate