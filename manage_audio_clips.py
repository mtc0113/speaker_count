import unsupervised_speaker_count
import os
import sys


# Move the control to Current Working Directory
path = os.getcwd()
print("\nLocation of the current python script:", path)
os.chdir(path)

# Find Total number of arguments passed to the script
n = len(sys.argv)
print("\nTotal number of arguments passed:", n)

# List the arguments passed
print("\nName of Python script:", sys.argv[0])

print("\nArguments passed:", end=" ")
for i in range(1, n):
    print(sys.argv[i], end=" ")

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

rev_mfcc_file = speech_folder_name + '/temp/rev.MFCC' + output_file_extension

merged_mfcc_file = speech_folder_name + '/temp/merged.MFCC' + output_file_extension


