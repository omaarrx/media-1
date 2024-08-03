import cv2
import os

# Create the folder to save the filtered frames
#os.makedirs('AvgFilter_ImgSeq', exist_ok=True)

# Read the video
video = cv2.VideoCapture('C:/Users/egypt2/Desktop/Assignment 2 Media/noisyvideo2.mp4')


import numpy as np

# Function to apply the 3x3 Average Filter
def apply_average_filter(frame):
    kernel = (1/9) * np.ones((3, 3), dtype=np.float32)
    return cv2.filter2D(frame, -1, kernel)

# Create a folder to save the filtered frames
output_folder = 'AvgFilter_ImgSeq'
os.makedirs(output_folder, exist_ok=True)

# Open the video file
video_path = '/content/noisyvideo2.mp4'
cap = cv2.VideoCapture(video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create an output video writer
output_path = os.path.join(output_folder, 'filtered_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process the video frames
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the Average Filter to the frame
    filtered_frame = apply_average_filter(frame)

    # Save the filtered frame
    output_frame_path = os.path.join(output_folder, f'filteredframe{frame_count}.jpg')
    cv2.imwrite(output_frame_path, filtered_frame)

    # Write the filtered frame to the output video
    output_video.write(filtered_frame)

    frame_count += 1

    if frame_count >= 10:
        break

# Release the video capture and output video writer
cap.release()
output_video.release()

   
# Get the video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Define the codec for the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create an output video writer
output = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# Initialize a counter for the saved frames
saved_frames = 0

# Loop through each frame
while video.isOpened() and saved_frames < 10:
    ret, frame = video.read()

    if not ret:
        break

    # Apply the 3x3 average filter
    filtered_frame = cv2.blur(frame, (3, 3))

    # Save the filtered frame as an image in the 'AvgFilter_ImgSeq' folder
    frame_path = f'AvgFilter_ImgSeq/frame_{saved_frames + 1}.jpg'
    cv2.imwrite(frame_path, filtered_frame)

    # Write the filtered frame to the output video
    output.write(filtered_frame)

    # Display the filtered frame
    cv2.imshow('Filtered Video', filtered_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment the counter for saved frames
    saved_frames += 1

# Release resources
video.release()
output.release()
cv2.destroyAllWindows()

# Create the folder to save the transformed frames
os.makedirs('PixelTransform_ImgSeq', exist_ok=True)

# Loop through the first 10 filtered frames
for i in range(1, 11):
    # Read the filtered frame
    frame_path = f'AvgFilter_ImgSeq/frame_{i}.jpg'
    filtered_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

    # Apply pixel point transformation
    transformed_frame = f(filtered_frame)

    # Save the transformed frame as an image in the 'PixelTransform_ImgSeq' folder
    transformed_path = f'PixelTransform_ImgSeq/frame_{i}.jpg'
    cv2.imwrite(transformed_path, transformed_frame)

    # Display the transformed frame
    cv2.imshow('Transformed Frame', transformed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()