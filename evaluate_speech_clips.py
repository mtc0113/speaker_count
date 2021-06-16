# -*- coding: utf-8 -*-
"""
Created on Sun May 30 01:04:57 2021

@author: Partha Sarathi Paul
@Organization: IIT Kharagpur, India
"""

import os
import glob
import sys
import unsupervised_speaker_count as usc
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

output_file_extension = ".txt"
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


def main():
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

            # Copy files, but preserve metadata (cp -p src dst)
            shutil.copy2(rel_file_path, temporary_directory_path)

            # Revise the copied file path
            revised_file_path = temporary_directory_path + f"/{file}"

            # extract the name of the file (without file extension)
            file_name = os.path.splitext(file)[0]
            # split file_name entries by '_' character
            items = file_name.split('_')
            audio_owner = items[0]
            audio_record_date = datetime.date(int(items[3]), int(items[1]),int(items[2])).strftime("%d %b %Y")
            audio_record_time = datetime.time(int(items[4]), int(items[5]),int(items[6])).strftime("%H:%M:%S")

            speech_duration = usc.find_clip_length(path=revised_file_path)

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
    print(file_metadata_List)

    serial = []
    audio_name = []
    clip_length = []
    recorder = []
    record_date = []
    record_time = []
    num_segments = []
    num_voiced = []
    num_merged = []
    speaker_count = []
    compute_time = []
    for t in file_metadata_List:
        serial.append(t[0])
        audio_name.append(t[1])
        clip_length.append(t[2])
        recorder.append(t[3])
        record_date.append(t[4])
        record_time.append(t[5])
        num_segments.append(t[6])
        num_voiced.append(t[7])
        num_merged.append(t[8])
        speaker_count.append(t[9])
        compute_time.append(t[10])

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(serial[1:], clip_length[1:], color='maroon',
            width=0.4)

    plt.xlabel(serial[0])
    plt.ylabel(clip_length[0])
    plt.title("Test Plot")
    plt.show()


# Using the special variable __name__
if __name__ == "__main__":
    main()