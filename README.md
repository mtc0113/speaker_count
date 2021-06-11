# Unsupervised Speaker Count
A standalone implementation of Crowd++ algorithm described in the paper titled "Crowd++: Unsupervised Speaker Count with Smartphones" by Chenren Xu et al.
Estimates #Speakers in a conversation recorded using a smartphone.
Follows an unsupervised approach.
To run the script, do the following:
Put your conversation audio clips in a directory.
Pass the complete path of the audio directory (as a string) as command line argument.

Sample execution command:
python unsupervised_speaker_count.py "$Speech Data Folder$" [$tuning parameter 1$ $tuning parameter 2$]
$tuning parameter 1$ and $tuning parameter 2$ are optional arguments depending on make and model of the smartphone hardware.
