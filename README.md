# Vehicle Counting and Speed Detection with YOLOv3 and SORT Tracking Algorithm
(counting_output.gif)

## Requirements
Google Drive and Google Colab

## Running the model
1. Upload all the repo files to Google Drive
2. Create a new folder 'videos' and save the input video data 
3. Provide input/output video paths in main.py 
4. Open the notebook.ipynb in Google Colab and run the cells in sequential order
5. Output video and .txt files containing info on speed and counting will be saved in the folder 'output'

## Remarks
1. main.py includes the implementations for the counting and speed detection solution.
2. Violation of collinearity condition in cross-ratio computation gives rise to some error in speed.
