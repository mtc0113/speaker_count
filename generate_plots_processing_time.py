# -*- coding: utf-8 -*-
"""
Script to plot Distribution of length of the clips,
their processing time and the calculated speaker count
by parsing a metadata file, which is generated
by the "evaluate_speech_clips.py script" for the purpose
from the audio clips found in an input speech folder

Created on Sun May 30 01:04:57 2021

@author: Partha Sarathi Paul
@Organization: IIT Kharagpur, India
"""

import os
import sys
import csv
from datetime import datetime
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
    sys.exit("\nFolder \"" + speech_folder_name + "\" not found. Check the Speech folder path.")

# temporary_directory = "temp"
# temporary_directory_path = os.path.join(speech_folder_name, temporary_directory)

file_extension = ".txt"
fig_extension = ".png"

# metadata_file = speech_folder_name + '/temp/MetaData' + file_extension
#
# if os.path.exists(metadata_file) is False:
#     sys.exit("\nFiles in folder \"" + speech_folder_name + "\" not processed. Process the clips first and try.")

# fig_file1 = speech_folder_name + '/temp/fig1' + fig_extension
# fig_file2 = speech_folder_name + '/temp/fig2' + fig_extension


def join_tuple(tuple1, tuple2):
    tuple3 = (tuple1[0],tuple1[1],round(tuple1[2]+tuple2[2],2),tuple1[3],tuple1[4],tuple1[5],tuple1[6]+tuple2[6],
                  tuple1[7]+tuple2[7],tuple1[8]+tuple2[8],max(tuple1[9],tuple2[9]),round(tuple1[10]+tuple2[10],2))
    return tuple3


def plot_processing_time(speech_folder_path, meta_file_extension, fig_file_extension, *argv):
    metadata_file = speech_folder_path + '/temp/MetaData' + meta_file_extension

    if os.path.exists(metadata_file) is False:
        sys.exit("\nFiles in folder \"" + speech_folder_path + "\" not processed. Process the clips first and try.")

    fig_file2 = speech_folder_path + '/temp/fig2' + fig_file_extension

    line_count = 0
    file_metadata_List = []
    with open(metadata_file, "r") as f:
        csv_reader = csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE)

        for row in csv_reader:
            x = row[:]
            if line_count == 0:
                file_metadata = (x[0].strip("'"), x[1].strip("'"), x[2].strip("'"), x[3].strip("'"), x[4].strip("'"),
                                    x[5].strip("'"), x[6].strip("'"), x[7].strip("'"), x[8].strip("'"), x[9].strip("'"),
                                        x[10].strip("'"))
                file_metadata_List.append(file_metadata)
                line_count += 1
                continue
            else:
                file_metadata = (int(x[0]), x[1].strip("'"), float(x[2]), x[3].strip("'"), x[4].strip("'"),
                                 x[5].strip("'"), int(x[6]), int(x[7]), int(x[8]), int(x[9]), float(x[10]))
                file_metadata_List.append(file_metadata)
                line_count += 1



    # for t in file_metadata_List:
    #     print(t)

    if 'datewise' in argv:
        reduced_file_metadata_List = [file_metadata_List[0]]
        i = 1
        j = 2
        while i <= len(file_metadata_List) - 2:
            while j <= len(file_metadata_List) - 1 and file_metadata_List[j][3] == file_metadata_List[i][3] \
                        and file_metadata_List[j][4] == file_metadata_List[i][4]:
                file_metadata_List[i] = join_tuple(file_metadata_List[i],file_metadata_List[j])
                j += 1
            reduced_file_metadata_List.append(file_metadata_List[i])
            i = j
            j += 1
        file_metadata_List = reduced_file_metadata_List

        # print()
        # print("The Revised List")
        # for t in file_metadata_List:
        #     print(t)


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


    # set width of bar
    barWidth = 0.25
    fig = plt.figure(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(serial[1:]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    # br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, clip_length[1:], color='r', width=barWidth, edgecolor='grey', label='Clip Length')
    plt.bar(br2, compute_time[1:], color='g', width=barWidth, edgecolor='grey', label='Process Time')
    plt.bar(br3, speaker_count[1:], color='b', width=barWidth, edgecolor='grey', label='Speaker')


    # Adding Xticks
    if 'datewise' in argv:
        plt.xlabel('Recording Date', fontweight='bold', fontsize=15)
        plt.ylabel('Clip Length/Process Time (sec), #Speakers', fontweight='bold', fontsize=15)
        plt.yscale("log")
        plt.xticks([r + barWidth for r in range(len(serial[1:]))], record_date[1:])
    else:
        plt.xlabel('Recording Date, Time', fontweight='bold', fontsize=15)
        plt.ylabel('Clip Length/Process Time (sec), #Speakers', fontweight='bold', fontsize=15)
        plt.xticks([r + barWidth for r in range(len(serial[1:]))], zip(record_date[1:], record_time[1:]))

    plt.legend()
    plt.title("Change in Distribution of Clip Length and Its processing time: " + file_metadata_List[1][3].upper())
    plt.savefig(fig_file2)


def main():
    plot_processing_time(speech_folder_name, file_extension, fig_extension)

# Using the special variable __name__
if __name__ == "__main__":
    main()