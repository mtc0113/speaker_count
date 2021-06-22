# -*- coding: utf-8 -*-
"""
Script to find various metadata connected with the audio
clips found in the input speech folder, to calculate
speaker count in these clips using unsupervised speaker
count module developed earlier and plot the results

Created on Sun May 30 01:04:57 2021

@author: Partha Sarathi Paul
@Organization: IIT Kharagpur, India
"""

import os
import glob
import sys
import unsupervised_speaker_count as usc
import generate_plots_processing_time as gpp
import generate_plots_segment_count as gps
import shutil
from time import process_time
from time import sleep
import datetime
import numpy as np
import matplotlib.pyplot as plt


# Move the control to Current Working Directory
path = os.getcwd()
os.chdir(path)

# Find Total number of arguments passed to the script
n = len(sys.argv)

if n < 2:
    exit("\nPlease include recorded clips in a directory and\n" + "pass the directory path as command line argument")

# Input Audio Details
speech_folder_name = sys.argv[1]

if os.path.exists(speech_folder_name) is False:
    sys.exit("\nFolder \"" + speech_folder_name + "\" not found. Check the Speech folder path")

# tuple of supported audio file extensions
file_extension = ('.mp3', '.wav')

# Temporary intermediate files for processing
temporary_directory = "tempdir"
temporary_directory_path = os.path.join(speech_folder_name, temporary_directory)
temporary_directory2 = "temp"
temporary_directory_path2 = os.path.join(speech_folder_name, temporary_directory2)

if os.path.exists(temporary_directory_path) is False:
    os.makedirs(temporary_directory_path)
else:
    shutil.rmtree(temporary_directory_path)
    sleep(1.0)
    os.makedirs(temporary_directory_path)

if os.path.exists(temporary_directory_path2) is False:
    os.makedirs(temporary_directory_path2)
else:
    shutil.rmtree(temporary_directory_path2)
    sleep(1.0)
    os.makedirs(temporary_directory_path2)

output_file_extension = ".txt"
fig_file_extension = ".png"
# meta_file_extension = ".csv"
metadata_file = speech_folder_name + '/temp/MetaData' + output_file_extension

yin_file = speech_folder_name + '/temp/YIN' + output_file_extension
mfcc_file = speech_folder_name + '/temp/MFCC' + output_file_extension
rev_mfcc_file = speech_folder_name + '/temp/rev.MFCC' + output_file_extension
merged_mfcc_file = speech_folder_name + '/temp/merged.MFCC' + output_file_extension

# SEGMENT_LENGTH = usc.SEGMENT_LENGTH

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def evaluate_speech_clips():
    if not os.listdir(speech_folder_name):
        sys.exit("No file in folder \"" + speech_folder_name + "\"")

    file_count = 0
    file_metadata_List = [('Serial', "Audio File Name", "Clip Length", "Speech Recorder", "Recording Date",
            "Recording Time", "#Segments", "#Voiced Segments", "#Merged Segments", "#Speakers", "Computation Time"),]

    for file in os.listdir(speech_folder_name):
        # Check the extension of the file
        if file.endswith(file_extension):
            file_count += 1
            rel_file_path = speech_folder_name + f"/{file}"

            print("File Count:", file_count, "File Name:", file)

            # Copy files, but preserve metadata (cp -p src dst)
            shutil.copy2(rel_file_path, temporary_directory_path)

            # Revise the copied file path
            revised_file_path = temporary_directory_path + f"/{file}"

            # extract the name of the file (without file extension)
            file_name = os.path.splitext(file)[0]
            # split file_name entries by '_' character
            items = file_name.split('_')
            audio_owner = items[0]
            audio_record_date = datetime.date(int(items[3]), int(items[2]),int(items[1])).strftime("%d %b %Y")
            audio_record_time = datetime.time(int(items[4]), int(items[5]),int(items[6])).strftime("%H:%M:%S")

            speech_duration = usc.find_clip_length(path=revised_file_path)
            print("Audio Length:", speech_duration)

            start = process_time()
            final_speaker_count, total_segments, total_voiced_segments, total_merged_segments = usc.count_speaker(
                temporary_directory_path, file_extension, yin_file, mfcc_file, rev_mfcc_file, merged_mfcc_file)
            end = process_time()
            computation_time = end - start

            file_metadata = (file_count, file, speech_duration, audio_owner, audio_record_date, audio_record_time) + \
                (total_segments, total_voiced_segments, total_merged_segments, final_speaker_count, computation_time)
            file_metadata_List.append(file_metadata)

            # remove the temporary file
            os.remove(revised_file_path)


    # remove the temporary directories
    os.rmdir(temporary_directory_path)
    filelist = glob.glob(os.path.join(temporary_directory_path2, "*"))
    for f in filelist:
        os.remove(f)
    sleep(1.0)

    # Generate the metadata file
    usc.file_write(file_metadata_List, metadata_file)
    sleep(2.0)
    if len(sys.argv) > 2:
        gps.plot_segment_count(speech_folder_name, output_file_extension, fig_file_extension, sys.argv[2])
        gpp.plot_processing_time(speech_folder_name, output_file_extension, fig_file_extension, sys.argv[2])
    else:
        gps.plot_segment_count(speech_folder_name, output_file_extension, fig_file_extension)
        gpp.plot_processing_time(speech_folder_name, output_file_extension, fig_file_extension)

# Using the special variable __name__
if __name__ == "__main__":
    evaluate_speech_clips()