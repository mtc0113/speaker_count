# Unsupervised Speaker Counting
A standalone implementation of Crowd++ algorithm described in the paper titled "Crowd++: Unsupervised Speaker Count with Smartphones" by Chenren Xu et al.
Estimates number of speakers in a conversation recorded using a smartphone.
Follows an unsupervised approach.
To run the script, do the following:
Put your conversation audio clips in a directory.
Pass the complete path of the audio directory (as a string) as command line argument.

Sample execution command:
"python unsupervised_speaker_count.py "$Speech Data Folder Name$" [$tuning parameter 1$ $tuning parameter 2$]"

$tuning parameter 1$ and $tuning parameter 2$ are optional arguments depending on make and model of the smartphone hardware.

# Evaluate Speech Clips
Script to evaluate the audio clips recorded using audio uploader smartphone application.
It parses the clips for finding audio recording metadata.
It calculates the speaker count in each audio clip in a given speech folder using the 'Unsupervised Speaker Count' script.
Finally, it plots the obtained metadata and other derived results for evaluation of the audio data and verify the scripts.

Sample execution command:
"python evaluate_speech_clips.py "$Speech Data Folder Name$" [$desired_plot_style$]"

$desired_plot_style$ is an optional argument to control the style of the resulting figure.
It has only one possible value: datewise
If the parameter is not supplied, then the script plots the results clip-wise.
If the parameter is supplied, then the script agregate the result datewise, and plot accordingly.
The above might give better visual when the number of clips are too high, and there are many recordings in a day.

# Semisupervised Speaker Counting
A standalone transcoding of semisupervised crowd counting program following crowd++ strategy.
Estimates number of speakers in a conversation recorded using a smartphone.
Using a given speech sample of a target speaker, estimates the level of participation of the target speaker in the conversation.

Sample execution command:
"python semisupervised_speaker_count.py "$Speech Data Folder Name$" "$Training Speech Sample Data Folder Name$"
