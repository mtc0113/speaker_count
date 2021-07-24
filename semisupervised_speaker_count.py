# -*- coding: utf-8 -*-
"""
Script to find the number of different speakers
identifiable in the set of audio clips found
in an input speech folder and to identify the
level of presence of a distinct speech signature
in the conversation

Created on Sun July 22 01:04:57 2021

@author: Partha Sarathi Paul
@Organization: IIT Kharagpur, India
"""

import os
import sys
from time import process_time
import librosa
import unsupervised_speaker_count as usc
import csv
import numpy as np
from statistics import mean

# Move the control to Current Working Directory
path = os.getcwd()
# print("\nLocation of the current python script:", path)
os.chdir(path)

# Find Total number of arguments passed to the script
n = len(sys.argv)

if n < 3:
    print("\nPut test conversation in a folder and pass the folder path as first command line argument")
    print("\nPut target speech signature in another folder and pass the folder path as second command line argument")
    exit(1)

# Input Audio Details
speech_folder_name = sys.argv[1]
calibration_folder_name = sys.argv[2]

if os.path.exists(speech_folder_name) is False:
    sys.exit("\nFolder \"" + speech_folder_name + "\" not found. Check the speech folder path")

if os.path.exists(calibration_folder_name) is False:
    sys.exit("\nFolder \"" + calibration_folder_name + "\" not found. Check the calibration folder path")

# tuple of supported audio file extensions
file_extension = ('.mp3', '.wav')

# Temporary intermediate files
temporary_directory = "temp"
tmp_tst_dir_path = os.path.join(speech_folder_name, temporary_directory)
tmp_cal_dir_path = os.path.join(calibration_folder_name, temporary_directory)

if os.path.exists(tmp_tst_dir_path) is False:
    os.makedirs(tmp_tst_dir_path)

if os.path.exists(tmp_cal_dir_path) is False:
    os.makedirs(tmp_cal_dir_path)

output_file_extension = ".txt"
tst_yin_file = speech_folder_name + '/temp/YIN' + output_file_extension
tst_mfcc_file = speech_folder_name + '/temp/MFCC' + output_file_extension
cal_yin_file = calibration_folder_name + '/temp/YIN' + output_file_extension
cal_mfcc_file = calibration_folder_name + '/temp/MFCC' + output_file_extension
rev_mfcc_file = speech_folder_name + '/temp/rev.MFCC' + output_file_extension
merged_mfcc_file = speech_folder_name + '/temp/merged.MFCC' + output_file_extension
rev_cal_mfcc_file = calibration_folder_name + '/temp/rev.MFCC' + output_file_extension
merged_cal_mfcc_file = calibration_folder_name + '/temp/merged.MFCC' + output_file_extension
new_mfcc_file = speech_folder_name + '/temp/new.MFCC' + output_file_extension

# Assumed Sampling Rate
# sampling_rate = 22050

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
MFCC_DIST_SAME_SEMI = 15.6
MFCC_DIST_DIFF_SEMI = 21.6

# Calibration Related Parameter
CAL_DURATION_SEC_LOWER = 45.0

# YIN Pitch Detection Method Parameters
frame_length = 512
hop_length = frame_length // 4
win_length = frame_length // 4

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


def cal_merge_segments(revised_ceptral_file, merged_ceptral_file):
    MFCC_List = []
    pitch_List = []

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
            pitch_List.append(curr_segment_pitch)
            mfcc_line_count += 1

    f.close()

    segment_pitch_mean = mean(pitch_List)

    # iteratively merge the neighbor segments until a single merged segment is formed
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
            # segment_pitch_q = item_q[2]
            segment_frame_count_q = item_q[3]
            mfcc_q = item_q[4:]

            # revised_audio_num_p = audio_num_p
            # revised_segment_num_p = segment_num_p
            # revised_segment_pitch_p = (segment_pitch_p + segment_pitch_q) / 2
            revised_segment_frame_count_p = segment_frame_count_p + segment_frame_count_q
            revised_mfcc_p = mfcc_p + mfcc_q

            revised_item_p = (audio_num_p, segment_num_p, segment_pitch_p,
                              revised_segment_frame_count_p) + tuple(revised_mfcc_p)
            MFCC_List[p] = revised_item_p

            MFCC_List.pop(q)
            p = q
            q += 1

        if last_size == len(MFCC_List):
            break

    # revise the pitch value with the mean pitch of the segment
    mfcc_tuple = MFCC_List[0]
    MFCC_List.pop(0)
    revised_mfcc_tuple = (0, 1) + (segment_pitch_mean,) + mfcc_tuple[3:]
    MFCC_List.append(revised_mfcc_tuple)

    usc.file_write(MFCC_List, merged_ceptral_file)
    # print("Size of Segment List after merging Matching Segments:", last_size, "\n")
    return MFCC_List


def self_calibration(cal_dir, audio_ext, cal_pitch_file, cal_cept_file, rev_cal_cept_file, merged_cal_cept_file):
    num_cal_segments, segment_frame_count = \
        usc.Generate_Feature_Files(cal_dir, audio_ext, cal_pitch_file, cal_cept_file)
    # Remove non-voiced audio segments from the list using generated Pitch and MFCC features
    num_cal_voiced_segments = \
        usc.Remove_Non_Voiced(cal_pitch_file, cal_cept_file, segment_frame_count, rev_cal_cept_file)
    print("Owner Speech Segments:", num_cal_segments, "Owner Voiced Segments:", num_cal_voiced_segments)

    if num_cal_voiced_segments * SEGMENT_LENGTH < CAL_DURATION_SEC_LOWER:
        print("Not Enough Calibration Data")
        return False

    cal_mfcc_list = cal_merge_segments(rev_cal_cept_file, merged_cal_cept_file)
    cal_mfcc_list_size = len(cal_mfcc_list)
    print("Calibration MFCC List Size:", cal_mfcc_list_size)

    return True


def semisupervised_speaker_counting(tst_cept_file, trn_cept_file):
    trn_feature_list = []
    tst_feature_list = []
    new_feature_list = []
    owner_presence_status = True

    # Extract the training speech features in the training feature list
    with open(trn_cept_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            curr_audio_num = int(row[0])
            curr_segment_num = int(row[1])
            curr_segment_pitch = float(row[2])
            curr_segment_frame_count = int(row[3])
            segment_mfcc_str = np.array(row[4:])
            segment_mfcc = segment_mfcc_str.astype(float)
            trn_feature_list.append(
                (curr_audio_num, curr_segment_num, curr_segment_pitch, curr_segment_frame_count) + tuple(segment_mfcc))
            line_count += 1
    f.close()

    # Extract the test audio features in the test feature list
    with open(tst_cept_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            curr_audio_num = int(row[0])
            curr_segment_num = int(row[1])
            curr_segment_pitch = float(row[2])
            curr_segment_frame_count = int(row[3])
            segment_mfcc_str = np.array(row[4:])
            segment_mfcc = segment_mfcc_str.astype(float)
            tst_feature_list.append(
                (curr_audio_num, curr_segment_num, curr_segment_pitch, curr_segment_frame_count) + tuple(segment_mfcc))
            line_count += 1
    f.close()

    trn_feature_list_size = len(trn_feature_list)
    tst_feature_list_size = len(tst_feature_list)
    print("Training Feature List Size:", trn_feature_list_size, "Test Feature List Size:", tst_feature_list_size)

    new_feature_list.append(trn_feature_list[0])
    speaker_count = 1
    distance = 0
    length = 0

    # new_audio_num = new_feature_list[0][0]
    # new_segment_num = new_feature_list[0][1]
    # new_pitch = new_feature_list[0][2]
    # new_frame_count = new_feature_list[0][3]
    # new_mfcc = new_feature_list[0][4:]

    for i in range(tst_feature_list_size):
        diff_count = 0
        pitch = tst_feature_list[i][2]
        frame_count = tst_feature_list[i][3]
        mfcc = tst_feature_list[i][4:]

        for j in range(speaker_count):
            print("i =", i, "j =", j, "New length:", new_feature_list[0][3], "length:", length,
                  "Distance:", distance, "Diff Count:", diff_count, "Current Speaker Count:", speaker_count)
            # for each segment i, compare it with each previously admitted segment j
            # pitch = tst_feature_list[i][2]
            # frame_count = tst_feature_list[i][3]
            # mfcc = tst_feature_list[i][4:]
            new_audio_num = new_feature_list[j][0]
            new_segment_num = new_feature_list[j][1]
            new_pitch = new_feature_list[j][2]
            new_frame_count = new_feature_list[j][3]
            new_mfcc = new_feature_list[j][4:]

            distance = usc.get_Distance(new_mfcc, mfcc)

            # different gender observed from pitch, so different speaker
            if usc.gender_decision(pitch, new_pitch) == 0:
                diff_count += 1
            # mfcc distance is larger than a threshold, so different speaker
            elif (j == 0 and distance >= MFCC_DIST_DIFF_SEMI) or (j > 0 and distance >= MFCC_DIST_DIFF_UN):
                diff_count += 1
            # same speaker
            else:
                if ((j == 0 and distance <= MFCC_DIST_SAME_SEMI) or (j > 0 and distance <= MFCC_DIST_SAME_UN)) \
                        and usc.gender_decision(pitch, new_pitch) == 1:
                    # Merge the segment
                    new_pitch = (new_pitch + pitch) / 2
                    new_frame_count = new_frame_count + frame_count
                    new_mfcc = new_mfcc + mfcc
                    new_item = (new_audio_num, new_segment_num, new_pitch, new_frame_count) + tuple(new_mfcc)
                    new_feature_list[j] = new_item
                    break

        # admit as a new speaker if different from all the admitted speakers
        if diff_count == speaker_count:
            speaker_count += 1
            # new_audio_num = tst_feature_list[i][0]
            # new_segment_num = tst_feature_list[i][1]
            # new_pitch = new_feature_list[i][2]
            # new_frame_count = new_feature_list[i][3]
            # new_mfcc = new_feature_list[i][4:]
            new_feature_list.append(tst_feature_list[i])

        # Revise the observed speech length using frame count of current test segment
        length += frame_count
        # length += tst_feature_list[i][3]
        # print("New length:", new_feature_list[0][3], "length:", length)

    # Do not count the owner if there is no voice from the owner in the conversation testing data
    if new_feature_list[0][3] == trn_feature_list[0][3]:
        speaker_count -= 1
        owner_presence_status = False

    owner_speech_percentage = 100 * (new_feature_list[0][3] - trn_feature_list[0][3]) / length

    # Store new feature list in a file
    usc.file_write(new_feature_list, new_mfcc_file)

    return speaker_count, owner_presence_status, owner_speech_percentage


def count_speaker_semi(speech_dir, cal_dir, audio_ext, tst_pitch_file, tst_cept_file, cal_pitch_file,
                       cal_cept_file, rev_cept_file, rev_cal_cept_file, merged_cept_file, merged_cal_cept_file):
    # Initialize
    speaker_count = 0
    owner_presence_status = False
    owner_speech_percentage = -1.0

    # Split each audio file (having supported extension) in speech folder into segments (of equal duration)
    # Generate Pitch and MFCC features for each segment
    # and save these feature vectors/matrices in corresponding feature files for further processing
    num_tst_segments, segment_frame_count = \
        usc.Generate_Feature_Files(speech_dir, audio_ext, tst_pitch_file, tst_cept_file)

    # Remove non-voiced audio segments from the list using generated Pitch and MFCC features
    num_voiced_segments = usc.Remove_Non_Voiced(tst_pitch_file, tst_cept_file, segment_frame_count, rev_cept_file)
    print("Total Test Segments:", num_tst_segments, "Voiced Test Segments:", num_voiced_segments)

    if num_voiced_segments == 0:
        print("No test voiced segment in Speech Folder")
        return speaker_count, owner_presence_status, owner_speech_percentage
    else:
        mfcc_list = usc.merge_segments(rev_cept_file, merged_cept_file)
        mfcc_list_size = len(mfcc_list)
        print("MFCC List Size:", mfcc_list_size)

    calibration = self_calibration(cal_dir, audio_ext, cal_pitch_file,
                                   cal_cept_file, rev_cal_cept_file, merged_cal_cept_file)

    if calibration is False:
        print("Calibration Failed due to insufficient owner voice sample")
        return speaker_count, owner_presence_status, owner_speech_percentage

    return semisupervised_speaker_counting(merged_cept_file, merged_cal_cept_file)


# Call the main function
def main_function():
    start = process_time()
    final_speaker_count, owner_status, owner_speech_percentage = \
        count_speaker_semi(speech_folder_name, calibration_folder_name, file_extension,
                           tst_yin_file, tst_mfcc_file, cal_yin_file, cal_mfcc_file,
                           rev_mfcc_file, rev_cal_mfcc_file, merged_mfcc_file, merged_cal_mfcc_file)
    end = process_time()
    print(final_speaker_count, "number of different speakers identified in all the audio clips of the input folder")
    if owner_status is False:
        print("Owner has not participated in the conversation")
    else:
        print("Owner Contribution in Test Conversation:", str(round(owner_speech_percentage,2)) + "%")
    print("Time elapsed during the computation:", end - start, "seconds")


# Using the special variable __name__
if __name__ == "__main__":
    main_function()
