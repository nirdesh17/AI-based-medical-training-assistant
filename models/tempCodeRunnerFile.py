annotation_1=pd.read_csv(os.environ.get('annotation_file1'),sep=' ',header=0)
annotation_2=pd.read_csv(os.environ.get('annotation_file2'),sep=' ',header=0)

prediction_1=pd.read_csv(os.environ.get('prediction_file1'),sep=' ',header=0)
prediction_2=pd.read_csv(os.environ.get('prediction_file2'),sep=' ',header=0)

#to capture the video file and get its frame rate and number of frames. 
#You can use these values to synchronize the timestamps of the annotation 
# and prediction files with the video frames
video_capture1 = cv2.VideoCapture(os.environ.get('video_file1'))
video_capture2 = cv2.VideoCapture(os.environ.get('video_file2'))

frame_rate1 = video_capture1.get(cv2.CAP_PROP_FPS)
frame_count1 = int(video_capture1.get(cv2.CAP_PROP_FRAME_COUNT))

frame_rate2 = video_capture2.get(cv2.CAP_PROP_FPS)
frame_count2 = int(video_capture2.get(cv2.CAP_PROP_FRAME_COUNT))