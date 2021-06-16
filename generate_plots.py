# -*- coding: utf-8 -*-
"""
Created on Sun May 30 01:04:57 2021

@author: Partha Sarathi Paul
@Organization: IIT Kharagpur, India
"""

import os
import sys


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

def main():
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

    # Make the plot
    plt.bar(br1, num_segments[1:], color='r', width=barWidth, edgecolor='grey', label='Initial')
    plt.bar(br2, num_voiced[1:], color='g', width=barWidth, edgecolor='grey', label='Voiced')
    plt.bar(br3, num_merged[1:], color='b', width=barWidth, edgecolor='grey', label='Merged')

    # Adding Xticks
    plt.xlabel('Audio File Name', fontweight='bold', fontsize=15)
    plt.ylabel('Segment Count', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(serial[1:]))], audio_name[1:])

    plt.legend()
    plt.show()
    # savefig('test.png')


# Using the special variable __name__
if __name__ == "__main__":
    main()

