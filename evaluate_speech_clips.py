# -*- coding: utf-8 -*-
"""
Created on Sun May 30 01:04:57 2021

@author: Partha Sarathi Paul
@Organization: IIT Kharagpur, India
"""

import os
import sys
import unsupervised_speaker_count as usc
import shutil
from time import process_time



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

# Temporary intermediate files
temporary_directory = "tempdir"
temporary_directory_path = os.path.join(speech_folder_name, temporary_directory)

if os.path.exists(temporary_directory_path) is False:
    os.makedirs(temporary_directory_path)

output_file_extension = ".txt"
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
    file_metadata_List = [['Serial', "Audio Name", "duration", "Owner", "dd", "mm", "yy", "hh", "mm", "ss",
                           "#segments", "#voiced", "#merged", "#speaker", "#time"], ]

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

            speech_duration = usc.find_clip_length(path=revised_file_path)

            start = process_time()
            final_speaker_count, total_segments, total_voiced_segments, total_merged_segments = usc.count_speaker(
                temporary_directory_path, file_extension, yin_file, mfcc_file, rev_mfcc_file, merged_mfcc_file)
            end = process_time()
            computation_time = end - start

            file_metadata = [file_count, file, speech_duration] + items + \
                [total_segments, total_voiced_segments, total_merged_segments, final_speaker_count, computation_time]
            file_metadata_List.append(file_metadata)

            # remove the temporary file
            os.remove(revised_file_path)

    usc.file_write(file_metadata_List, metadata_file)

    # remove the temporary directory
    os.rmdir(temporary_directory_path)


# Using the special variable __name__
if __name__ == "__main__":
    main()