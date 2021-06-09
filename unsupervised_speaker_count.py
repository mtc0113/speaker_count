# -*- coding: utf-8 -*-
"""
Created on Sun May 30 01:04:57 2021

@author: mtc01
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

# from scipy.spatial import distance

# Move the control to Current Working Directory
# path = "C:/Users/mtc01/My Working Folder/My Python Projects"

path = os.getcwd()
print(path, '\n')
os.chdir(path)

# Input Audio Details
speech_folder_name = "Sample Conversation Files"

if os.path.exists(speech_folder_name) == False:
    sys.exit("Folder \"" + speech_folder_name + "\" not found. Check the Speech folder path")

file_extension = ('.mp3', '.wav')  # tuple of supported audio file extensions

output_file_extension = ".txt"
yin_file = speech_folder_name + '/YIN' + output_file_extension
mfcc_file = speech_folder_name + '/MFCC' + output_file_extension

# rev_yin_file = speech_folder_name + '/rev.YIN' + output_file_extension
rev_mfcc_file = speech_folder_name + '/rev.MFCC' + output_file_extension

merged_mfcc_file = speech_folder_name + '/merged.MFCC' + output_file_extension

# Selected Audio Signal Parameters
sr = 22050

SEGMENT_LENGTH = 3.0  # measured in second

PITCH_MALE_UPPER = 160  # measured in Hertz
PITCH_FEMALE_LOWER = 190  # measured in Hertz
PITCH_HUMAN_UPPER = 450  # measured in Hertz
PITCH_HUMAN_LOWER = 50  # measured in Hertz

PITCH_RATE_LOWER = 0.05  # Adopted from crowdpp Android Implementation
PITCH_MU_LOWER = 50
PITCH_MU_UPPER = 450
PITCH_SIGMA_UPPER = 100

MFCC_DIST_SAME_UN = 21.6  # Adopted from crowdpp Android Implementation

# Selected YIN Signal Processing Parameters
frame_length = 1024
hop_length = frame_length // 8
win_length = frame_length // 8

fmin = 10
fmax = 2093
trough_threshold = 0.1

# Selected MFCC Signal Processing Parameters
n_fft = frame_length
n_mfcc = 20


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


# Function to Derive YIN and MFCC features for the Input Clip
def derive_features(file_count, filename, segment_length):
    YIN = []
    MFCC = []

    current_speech, sr = librosa.load(filename)
    duration = librosa.get_duration(y=current_speech, sr=sr)

    print('')
    print("Audio File Name:", filename)
    print("Audio Clip Duration:", duration)
    print('')

    start_segment = 0.0
    end_segment = start_segment + segment_length
    segment_count = 0
    while end_segment <= duration:
        # Load a speech segment of duration 'segment_length' from the input audio clip
        speech_segment, sr = librosa.load(filename, sr=sr, mono=True, offset=start_segment, duration=segment_length)
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
        # print("MFCC Frames:",len(segment_mfcc))
        segment_mfcc_tr = segment_mfcc.transpose()
        # print("MFCC Transpose Frames:",len(segment_mfcc_tr))

        mfcc_tuple = ()
        frame_count = len(segment_mfcc_tr)
        for i in range(frame_count):
            mfcc_tuple = mfcc_tuple + tuple(segment_mfcc_tr[i])

        MFCC.append((file_count, segment_count) + mfcc_tuple)

        start_segment = end_segment
        end_segment = start_segment + segment_length

    return YIN, MFCC, frame_count


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
    YIN_List = []
    MFCC_List = []

    if os.listdir(folder_name) == []:
        sys.exit("No file in folder \"" + speech_folder_name + "\"")

    file_count = 0
    for file in os.listdir(folder_name):
        # Check the extension of the file
        if file.endswith(extension):
            rel_file_path = folder_name + f"/{file}"
            file_count += 1
            this_YIN_List, this_MFCC_List, frame_count = derive_features(file_count, rel_file_path, SEGMENT_LENGTH)
            YIN_List = YIN_List + this_YIN_List
            MFCC_List = MFCC_List + this_MFCC_List

    if file_count == 0:
        sys.exit("No files in folder \"" + speech_folder_name + "\" has supported audio file extesion")

    file_write(YIN_List, outfile_1)
    file_write(MFCC_List, outfile_2)

    print(file_count, "supported Audio Files in folder \"" + speech_folder_name + "\"", "\n")

    return frame_count


# Function to Remove Non-voiced Segments

def Remove_Non_Voiced(in_file1, in_file2, out_file, frame_count):
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

            pitch_rate = c / len(segment_pitch)
            pitch_mu = statistics.mean(temp_pitch)
            pitch_sigma = statistics.stdev(temp_pitch)

            if pitch_rate >= PITCH_RATE_LOWER and pitch_mu >= PITCH_MU_LOWER and pitch_mu <= PITCH_MU_UPPER and pitch_sigma <= PITCH_SIGMA_UPPER:
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

            if revised_YIN_List[mfcc_line_count][0] == curr_audio_num and revised_YIN_List[mfcc_line_count][
                1] == curr_segment_num:
                revised_MFCC_List.append((curr_audio_num, curr_segment_num, float(revised_YIN_List[mfcc_line_count][2]),
                                          frame_count) + tuple(segment_mfcc))
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

    # for i in range(n_fft):
    #     column = []
    #     for j in range(num_frames):
    #         column.append(mfcc[i + j * n_fft])

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
    print("Radian Distance:", radian_distance)

    degree_distance = math.degrees(radian_distance)
    print("Degree Distance:", degree_distance)

    return degree_distance


def merge_segments(pitch_file, ceptral_file, revised_ceptral_file, merged_ceptral_file):
    segment_frame_count = Generate_Feature_Files(speech_folder_name, file_extension, pitch_file, ceptral_file)

    voice_count = Remove_Non_Voiced(pitch_file, ceptral_file, revised_ceptral_file, segment_frame_count)

    if voice_count == 0:
        sys.exit("No Voiced Segment in Folder" + speech_folder_name)
    else:
        print(voice_count, "number of voiced segments found in folder \"" + speech_folder_name + "\"\n")

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

            # print(len(segment_mfcc))

        if mfcc_line_count == voice_count:
            print("File Copy OK\n")

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

            print("p:", p, "q:", q, "Cosine Distance:", distance)

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
    print("Last Size:", last_size, "\n")


# The main function for Unsupervised Speaker Counting from a given set of speech files

def main_function():
    merge_segments(yin_file, mfcc_file, rev_mfcc_file, merged_mfcc_file)


# Call of the main function

main_function()



