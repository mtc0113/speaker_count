# -*- coding: utf-8 -*-
"""
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
metadata_file = speech_folder_name + '/temp/MetaData' + file_extension

if os.path.exists(metadata_file) is False:
    sys.exit("\nFiles in folder \"" + speech_folder_name + "\" not processed. Process the clips first and try.")

fig_extension = ".png"
fig_file1 = speech_folder_name + '/temp/fig1' + fig_extension
# fig_file2 = speech_folder_name + '/temp/fig2' + fig_extension

def main():
    line_count = 0
    file_metadata_List = []
    with open(metadata_file, "r") as f:
        csv_reader = csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE)

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                file_metadata = row[:]
                file_metadata_List.append(file_metadata)
                line_count += 1




    # print(file_metadata_List)

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

    print()
    serial = list(map(int, serial))
    print(serial)
    audio_name = [x.strip("'") for x in audio_name]
    print(audio_name)
    clip_length = list(map(float, clip_length))
    print(clip_length)
    recorder = [x.strip("'") for x in recorder]
    print(recorder)
    record_date = [datetime.strptime(x, "'%d %b %Y'").strftime("%d %b %Y") for x in record_date]
    print(record_date)
    record_time = [datetime.strptime(x, "'%H:%M:%S'").strftime('%H:%M:%S') for x in record_time]
    print(record_time)
    num_segments = list(map(int, num_segments))
    print(num_segments)
    num_voiced = list(map(int, num_voiced))
    print(num_voiced)
    num_merged = list(map(int, num_merged))
    print(num_merged)
    speaker_count = list(map(int, speaker_count))
    print(speaker_count)
    compute_time = list(map(float, compute_time))
    print(compute_time)
    print(clip_length)


    # set width of bar
    barWidth = 0.15
    fig = plt.figure(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(serial))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, num_segments, color='r', width=barWidth, edgecolor='grey', label='Initial')
    plt.bar(br2, num_voiced, color='g', width=barWidth, edgecolor='grey', label='Voiced')
    plt.bar(br3, num_merged, color='b', width=barWidth, edgecolor='grey', label='Merged')
    plt.bar(br4, speaker_count, color='m', width=barWidth, edgecolor='grey', label='Speaker')

    # Adding Xticks
    plt.xlabel('Date, Time', fontweight='bold', fontsize=15)
    plt.ylabel('Segment/Speaker Count', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(serial))], zip(record_date, record_time))

    plt.legend()
    plt.title("Change in Distribution of Audio Segments and Speaker Counts: Soumyajit")


    plt.savefig(fig_file1)


# Using the special variable __name__
if __name__ == "__main__":
    main()

