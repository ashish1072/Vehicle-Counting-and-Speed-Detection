# Vehicle Counting and Speed Detection with YOLOv3 and SORT

## Requirements
Google Drive and Google Colab

## Running the model
1. Upload all the files & folders in the repository, to Google Drive
2. Put the files to be processed in 'videos' folder
3. Provide input/output video paths in main.py 
4. Open the notebook.ipynb in Google Colab and run the cells in sequential order
5. Output video and text files will be saved in the folder 'output'

## Remarks
1. main.py is the most important file with comments whereever necessary to improve/modify the counting or speed detection solution.
2. During speed estimation, sometimes there could be unrealistic values due to the cars entering or leaving the parking area very close to the virtual speed detection line. 
3. Violation of collinearity in cross-ratio computation gives rise to some error in speed.


