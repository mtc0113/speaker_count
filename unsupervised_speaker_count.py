# -*- coding: utf-8 -*-
"""
Script to find the number of different speakers
identifiable in the set of audio clips found
in an input speech folder

Created on Sun May 30 01:04:57 2021

@author: Partha Sarathi Paul
@Organization: IIT Kharagpur, India
"""

import os
import sys
import numpy as np
import librosa
import librosa.display
import statistics
from numpy import linalg as la
import csv
import math
from time import process_time

# Move the control to Current Working Directory
path = os.getcwd()
# print("\nLocation of the current python script:", path)
os.chdir(path)

# Find Total number of arguments passed to the script
n = len(sys.argv)
# print("\nTotal number of arguments passed:", n)

# List the arguments passed
# print("\nName of Python script:", sys.argv[0])

# print("\nArguments passed:", end=" ")
# for i in range(1, n):
#     print(sys.argv[i], end=" ")
# print()

if n < 2:
    exit("\nPlease include recorded clips in a directory and\n" + "pass the directory path as command line argument")

# Input Audio Details
speech_folder_name = sys.argv[1]

if os.path.exists(speech_folder_name) is False:
    sys.exit("\nFolder \"" + speech_folder_name + "\" not found. Check the Speech folder path")

# tuple of supported audio file extensions
file_extension = ('.mp3', '.wav')

# Temporary intermediate files
temporary_directory = "temp"
temporary_directory_path = os.path.join(speech_folder_name, temporary_directory)

if os.path.exists(temporary_directory_path) is False:
    os.makedirs(temporary_directory_path)

output_file_extension = ".txt"
yin_file = speech_folder_name + '/temp/YIN' + output_file_extension
mfcc_file = speech_folder_name + '/temp/MFCC' + output_file_extension
# metadata_file = speech_folder_name + '/temp/MetaData' + output_file_extension

rev_mfcc_file = speech_folder_name + '/temp/rev.MFCC' + output_file_extension

merged_mfcc_file = speech_folder_name + '/temp/merged.MFCC' + output_file_extension

# Assumed Sampling Rate
# sr = 22050

# Application Parameters: Adopted from crowdpp Android Implementation
SEGMENT_LENGTH = 3.0  # measured in second

PITCH_MALE_UPPER = 160  # measured in Hertz
PITCH_FEMALE_LOWER = 190  # measured in Hertz
PITCH_HUMAN_UPPER = 450  # measured in Hertz
PITCH_HUMAN_LOWER = 50  # measured in Hertz

PITCH_RATE_LOWER = 0.05
PITCH_MU_LOWER = 50  # measured in Hertz
PITCH_MU_UPPER = 450  # measured in Hertz
PITCH_SIGMA_UPPER = 100  # measured in Hertz

# Default Tuning Parameters: Adopted from crowdpp Android Implementation
MFCC_DIST_SAME_UN = 15.6
MFCC_DIST_DIFF_UN = 21.6

# Set Tuning Parameters using command line arguments to the script
if n > 3:
    MFCC_DIST_SAME_UN = float(sys.argv[2])
    MFCC_DIST_DIFF_UN = float(sys.argv[3])

# YIN Pitch Detection Method Parameters
frame_length = 1024
hop_length = frame_length // 8
win_length = frame_length // 8

fmin = 10
fmax = 2093
trough_threshold = 0.1

# MFCC Signal Processing Parameters
n_fft = frame_length
n_mfcc = 20

# Suppress repeated "PySoundFile failed. Trying audioread instead." warning by Librosa
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


# Estimate Gender from Pitch feature
def estimate_gender(pitch):
    gender = -1  # Gender Uncertain
    if pitch <= PITCH_MALE_UPPER:
        gender = 0  # Male
    elif pitch >= PITCH_FEMALE_LOWER:
        gender = 1  # Female
    return gender


# Decide on Gender Similarity based on Pitch feature
def gender_decision(pitch_a, pitch_b):
    gender_a = estimate_gender(pitch_a)
    gender_b = estimate_gender(pitch_b)

    if gender_a != -1 and gender_b != -1:  # Clear gender identification through Pitch feature
        if gender_a == gender_b:
            return 1  # Same gender
        else:
            return 0  # Different gender
    else:
        return -1  # leave the job to MFCC


def find_clip_length(path):
    current_audio, sr = librosa.load(path)
    audio_duration = librosa.get_duration(y=current_audio, sr=sr)
    return audio_duration

# Function to Derive YIN and MFCC features of a Segment
def derive_features(file_count, filename, segment_length):
    YIN = []
    MFCC = []
    frame_count = 0

    duration = find_clip_length(path=filename)

    # print("Audio Clip", file_count, ":", filename)
    # print("Audio Clip Duration:", duration)

    start_segment = 0.0
    end_segment = start_segment + segment_length
    segment_count = 0
    while end_segment <= duration:
        # Load a speech segment of duration 'segment_length' from the input audio clip
        speech_segment, sr = librosa.load(filename, mono=True, offset=start_segment, duration=segment_length)
        segment_count += 1

        # Calculate average fundamental frequency of the segment
        f0 = librosa.yin(y=speech_segment, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length,
                         win_length=win_length, hop_length=hop_length, trough_threshold=trough_threshold)
        # print("YIN Frames:",len(f0))

        # for x in f0:
        pitch_tuple = (file_count, segment_count) + tuple(f0)

        YIN.append(pitch_tuple)

        # Calculate MFCCs for the segment
        segment_mfcc = librosa.feature.mfcc(speech_segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc,
                                            win_length=win_length)
        # Deselect the first coefficient for not modeling DC component of the audio signal (as per crowd++ paper)
        segment_mfcc_select = segment_mfcc[1:]
        # print("MFCC Frames:",len(segment_mfcc))
        print(segment_mfcc_select.shape)
        segment_mfcc_tr = segment_mfcc_select.transpose()
        # print("MFCC Transpose Frames:",len(segment_mfcc_tr))

        mfcc_tuple = ()
        frame_count = len(segment_mfcc_tr)
        for i in range(frame_count):
            mfcc_tuple = mfcc_tuple + tuple(segment_mfcc_tr[i])

        MFCC.append((file_count, segment_count) + mfcc_tuple)

        start_segment = end_segment
        end_segment = start_segment + segment_length

    return duration, segment_count, frame_count, YIN, MFCC


# Function to write the feature vectors to feature file
def file_write(feature_List, output_file):
    with open(output_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f

        for x in feature_List:
            print(str(x)[1:-1])

        sys.stdout = original_stdout
        f.close()


# Function to Generate Feature Vectors for the Speech Audio Clips in the Input Directory
def Generate_Feature_Files(folder_name, extension, outfile_1, outfile_2):
    # file_metadata_List = []
    YIN_List = []
    MFCC_List = []

    if not os.listdir(folder_name):
        sys.exit("No file in folder \"" + speech_folder_name + "\"")

    file_count = 0
    tot_segments = 0
    segment_count = 0
    frame_count = 0
    for file in os.listdir(folder_name):
        # Check the extension of the file
        if file.endswith(extension):
            rel_file_path = folder_name + f"/{file}"
            # extract the name of the file (without file extension)
            # file_name = os.path.splitext(file)[0]
            # # split file_name entries by '_' character
            # items = file_name.split('_')

            file_count += 1
            speech_duration, segment_count, frame_count, this_YIN_List, this_MFCC_List = \
                derive_features(file_count, rel_file_path, SEGMENT_LENGTH)
            YIN_List = YIN_List + this_YIN_List
            MFCC_List = MFCC_List + this_MFCC_List
            tot_segments += segment_count

            # file_metadata = [file_count, file, speech_duration, segment_count] + items
            # file_metadata_List.append(file_metadata)

    if file_count == 0:
        sys.exit("No files in folder \"" + speech_folder_name + "\" has supported audio file extesion")

    file_write(YIN_List, outfile_1)
    file_write(MFCC_List, outfile_2)
    # file_write(file_metadata_List, meta_info_file)

    # print(file_count, "supported Audio Clips in folder \"" + speech_folder_name + "\"", "\n")

    return tot_segments, frame_count


# Function to Remove Non-voiced Segments
def Remove_Non_Voiced(in_file1, in_file2, frame_count, out_file):
    with open(in_file1, "r") as f:
        csv_reader = csv.reader(f, delimiter=',')

        line_count = 0
        revised_YIN_List = []

        for row in csv_reader:
            curr_audio_num = int(row[0])
            curr_segment_num = int(row[1])
            segment_pitch = row[2:]

            c = 0
            temp_pitch = []
            for curr_pitch in segment_pitch:
                if float(curr_pitch) >= PITCH_HUMAN_LOWER and float(curr_pitch) <= PITCH_HUMAN_UPPER:
                    c += 1
                    temp_pitch.append(float(curr_pitch))
            if len(temp_pitch) != 0:
                pitch_rate = c / len(segment_pitch)
                pitch_mu = statistics.mean(temp_pitch)
                pitch_sigma = statistics.stdev(temp_pitch)
                # print("Audio Num:", curr_audio_num, "Segment Num:", curr_segment_num, "Status: Voiced")
            else:
                pitch_rate = 0.0
                pitch_mu = 0.0
                pitch_sigma = 0.0
                # print("Audio Num:", curr_audio_num, "Segment Num:", curr_segment_num, "Status: Non-voiced")

            if pitch_rate >= PITCH_RATE_LOWER and \
                    pitch_mu >= PITCH_MU_LOWER and pitch_mu <= PITCH_MU_UPPER and pitch_sigma <= PITCH_SIGMA_UPPER:
                revised_YIN_List.append((curr_audio_num, curr_segment_num, pitch_mu))
                line_count += 1

        if line_count == 0:
            return line_count

        # file_write(revised_YIN_List, out_file1)

    f.close()

    with open(in_file2, "r") as f:
        csv_reader = csv.reader(f, delimiter=',')

        mfcc_line_count = 0
        revised_MFCC_List = []

        for row in csv_reader:
            curr_audio_num = int(row[0])
            curr_segment_num = int(row[1])
            segment_mfcc_str = np.array(row[2:])
            segment_mfcc = segment_mfcc_str.astype(float)

            if revised_YIN_List[mfcc_line_count][0] == curr_audio_num and \
                    revised_YIN_List[mfcc_line_count][1] == curr_segment_num:
                revised_MFCC_List.append((curr_audio_num, curr_segment_num, float(
                    revised_YIN_List[mfcc_line_count][2]), frame_count) + tuple(segment_mfcc))
                mfcc_line_count += 1

            if mfcc_line_count == line_count:
                break

        file_write(revised_MFCC_List, out_file)

    f.close()
    return line_count


# Function to colculate the column mean of an MFCC matrix
# Order of MFCC representation is (num_frames,n_fft)
# MFCC is represented as an 1-D array obtained by appending
# 'num_frame' rows of size 'n_fft' placed side-by-side
def get_column_mean(mfcc):
    column_mean_arr = []

    for i in range(n_mfcc):
        column = []
        for j in range(i, len(mfcc), n_mfcc):
            column.append(mfcc[j])

        # print(len(column))
        column_mean = np.mean(column)
        column_mean_arr.append(column_mean)

    return column_mean_arr


# Function to find Distance (in degrees) between input MFCC matrices
def get_Distance(mfcc1, mfcc2):
    # Get aveage MFCC vector for the segment pair using column mean of the input matrices
    mfcc_a = get_column_mean(mfcc1)
    mfcc_b = get_column_mean(mfcc2)

    cosine_distance = np.dot(mfcc_a, mfcc_b) / (la.norm(mfcc_a) * la.norm(mfcc_b))
    # cosine_distance = distance.cosine(mfcc_a, mfcc_b)

    radian_distance = math.acos(cosine_distance)
    # print("Radian Distance:", radian_distance)

    degree_distance = math.degrees(radian_distance)
    # print("Degree Distance:", degree_distance)

    return degree_distance


# Function to merge matching neighbor segments
def merge_segments(revised_ceptral_file, merged_ceptral_file):
    MFCC_List = []

    with open(revised_ceptral_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=',')

        mfcc_line_count = 0

        for row in csv_reader:
            curr_audio_num = int(row[0])
            curr_segment_num = int(row[1])
            curr_segment_pitch = float(row[2])
            curr_segment_frame_count = int(row[3])
            segment_mfcc_str = np.array(row[4:])
            segment_mfcc = segment_mfcc_str.astype(float)

            MFCC_List.append(
                (curr_audio_num, curr_segment_num, curr_segment_pitch, curr_segment_frame_count) + tuple(segment_mfcc))
            mfcc_line_count += 1

    f.close()

    # iteratively pre-cluster the neighbor segments until no merging happens
    while True:
        last_size = len(MFCC_List)
        p = 0
        q = 1

        while q < len(MFCC_List):
            item_p = MFCC_List[p]
            audio_num_p = item_p[0]
            segment_num_p = item_p[1]
            segment_pitch_p = item_p[2]
            segment_frame_count_p = item_p[3]
            mfcc_p = item_p[4:]

            item_q = MFCC_List[q]
            # audio_num_q = item_q[0]
            # segment_num_q = item_q[1]
            segment_pitch_q = item_q[2]
            segment_frame_count_q = item_q[3]
            mfcc_q = item_q[4:]

            distance = get_Distance(mfcc_p, mfcc_q)
            decision = gender_decision(segment_pitch_p, segment_pitch_q)

            # print("p:", p, "q:", q,"Cosine Distance:", distance)
            if distance <= MFCC_DIST_SAME_UN and decision == 1:
                revised_audio_num_p = audio_num_p
                revised_segment_num_p = segment_num_p
                revised_segment_pitch_p = (segment_pitch_p + segment_pitch_q) / 2
                revised_segment_frame_count_p = segment_frame_count_p + segment_frame_count_q
                revised_mfcc_p = mfcc_p + mfcc_q

                revised_item_p = (revised_audio_num_p, revised_segment_num_p, revised_segment_pitch_p,
                                  revised_segment_frame_count_p) + tuple(revised_mfcc_p)
                MFCC_List[p] = revised_item_p

                MFCC_List.pop(q)
            else:
                p = q
                q += 1

        if last_size == len(MFCC_List):
            break

    file_write(MFCC_List, merged_ceptral_file)
    # print("Size of Segment List after merging Matching Segments:", last_size, "\n")
    return MFCC_List


# The main function for Unsupervised Speaker Counting from a given set of speech files
def count_speaker(speech_folder, audio_extension, pitch_file, cept_file, rev_cept_file, merged_cept_file):
    # Split each audio file (having supported extension) in speech folder into segments (of equal duration)
    # Generate Pitch and MFCC features for each segment
    # and save these feature vectors/matrices in corresponding feature files for further processing
    num_segments, segment_frame_count = \
        Generate_Feature_Files(speech_folder, audio_extension, pitch_file, cept_file)

    # Remove non-voiced audio segments from the list using generated Pitch and MFCC features
    num_voiced_segments = Remove_Non_Voiced(pitch_file, cept_file, segment_frame_count, rev_cept_file)

    print("Segment:", num_segments, "Voiced Segments:", num_voiced_segments)

    if num_voiced_segments == 0:
        speaker_count = 0
        mfcc_list_size = 0
        return speaker_count, num_segments, num_voiced_segments, mfcc_list_size
    else:
        # admit the first segment as speaker 1
        speaker_count = 1
    #     print(num_voiced_segments, "number of voiced segments found in folder \"" + speech_folder_name + "\"\n")

    # Iteratively merge matching neighboring voice segments in the list
    mfcc_list = merge_segments(rev_cept_file, merged_cept_file)
    mfcc_list_size = len(mfcc_list)

    new_audio_num = mfcc_list[0][0]
    new_segment_num = mfcc_list[0][1]
    new_pitch = mfcc_list[0][2]
    new_frame_count = mfcc_list[0][3]
    new_mfcc = mfcc_list[0][4:]

    # print("\n")
    for i in range(1, mfcc_list_size):
        diff_count = 0
        for j in range(speaker_count):
            # print("i =", i, "j =", j, "List Size =", len(mfcc_list), "Current Speaker Count:", speaker_count)
            # for each segment i, compare it with each previously admitted segment j
            pitch = mfcc_list[i][2]
            frame_count = mfcc_list[i][3]
            mfcc = mfcc_list[i][4:]

            distance = get_Distance(new_mfcc, mfcc)

            if gender_decision(pitch, new_pitch) == 0:  # different gender observed from pitch, so different speaker
                diff_count += 1
            elif distance >= MFCC_DIST_DIFF_UN:  # mfcc distance is larger than a threshold, so different speaker
                diff_count += 1
            else:  # mfcc distance is larger than a threshold, so different speaker
                if gender_decision(pitch, new_pitch) == 1 and distance <= MFCC_DIST_SAME_UN:
                    # Merge the segment
                    new_pitch = (new_pitch + pitch) / 2
                    new_frame_count = new_frame_count + frame_count
                    new_mfcc = new_mfcc + mfcc

                    new_item = (new_audio_num, new_segment_num, new_pitch, new_frame_count) + tuple(new_mfcc)
                    mfcc_list[j] = new_item

        # admit as a new speaker if different from all the admitted speakers
        if diff_count == speaker_count:
            speaker_count += 1
            new_audio_num = mfcc_list[i][0]
            new_segment_num = mfcc_list[i][1]
            new_pitch = mfcc_list[i][2]
            new_frame_count = mfcc_list[i][3]
            new_mfcc = mfcc_list[i][4:]

    return speaker_count, num_segments, num_voiced_segments, mfcc_list_size


# Call the main function
def main():
    start = process_time()
    final_speaker_count, total_segments, total_voiced_segments, total_merged_segments = count_speaker(
        speech_folder_name, file_extension, yin_file, mfcc_file, rev_mfcc_file, merged_mfcc_file)
    end = process_time()

    # print()
    print("Total number of segments processed:", total_segments)
    print("Total number of voiced segments identified:", total_voiced_segments)
    print("Total number of segments after the merger of neighboring voiced segments:", total_merged_segments)
    print(final_speaker_count, "number of different speakers identified in all the audio clips of the input folder")
    print("Time elapsed during the computation:", end - start, "seconds")


# Using the special variable __name__
if __name__ == "__main__":
    main()
