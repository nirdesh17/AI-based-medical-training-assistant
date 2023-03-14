
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load annotation file
ann_file = r"D:\SAP\CapStone Projects\data\anno.txt"
with open(ann_file, "r") as f:
    annotations = f.readlines()[1:] 

tool_names = set([line.split()[1] for line in annotations])
tool_names = list(set(tool_names))
num_tools = len(tool_names)

# Load prediction file
pred_file = r"D:\SAP\CapStone Projects\data\pred.txt"
with open(pred_file, "r") as f:
    predictions = f.readlines()[1:] 

# Load video file
video_file = r"D:\SAP\CapStone Projects\data\tool_video_02_Trim.mp4"
cap = cv2.VideoCapture(video_file)

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(480, 640, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # layers.Dense(1, activation='sigmoid')
    layers.Dense(num_tools, activation='softmax')

])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
for epoch in range(10):
    for i in range(len(annotations)):
        # Load annotation for the current frame
        annotation = annotations[i].split()
        frame_num = int(annotation[0])
        tool_present = int(annotation[1])

        # Load prediction for the current frame
        prediction = predictions[i].split()
        if len(prediction) != 5:
            continue
        x, y, w, h = map(int, prediction[1:])
        tool_bbox = np.array([x, y, x + w, y + h])

        # Load and preprocess the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (640, 480))
        frame = frame / 255.0

        # Train the model on the current frame
        tool_labels = np.array([int(x) for x in annotation[2:]])
        # model.train_on_batch(np.array([frame]), np.array([tool_present]))
        model.train_on_batch(np.array([frame]), np.array([tool_labels]))


# Process each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    frame = cv2.resize(frame, (640, 480))
    frame = frame / 255.0

    # Predict tool presence on the frame
    tool_present = model.predict(np.array([frame]))
    tool_probabilities = model.predict(np.array([frame]))[0]
    tool_index = np.argmax(tool_probabilities)
    tool_name = tool_names[tool_index]
    if tool_present.any() > 0.5:
        # cv2.putText(frame, "Tool detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tool name: {tool_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



    # Display the resulting frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
