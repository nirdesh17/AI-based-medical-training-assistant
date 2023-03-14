import numpy as np

# read ground truth annotations
gt_file = r"D:\SAP\CapStone Projects\data\train_dataset\tool_video_01.txt"
gt_data = np.genfromtxt(gt_file, delimiter='\t', skip_header=1)
gt_data = gt_data.reshape((-1, 8))
gt_frames = gt_data[:,0]
gt_labels = gt_data[:,1:]

# read prediction file
pred_file = r"D:\SAP\CapStone Projects\data\train_dataset\tool_video_01_pred.txt"
pred_data = np.genfromtxt(pred_file, delimiter='\t', skip_header=1)
pred_data = pred_data.reshape((-1, 8))
pred_frames = pred_data[:,0]
pred_labels = pred_data[:,1:]

# compute true positive and false positive rates
n_gt = len(gt_frames)
n_pred = len(pred_frames)
tp = np.zeros(n_pred)
fp = np.zeros(n_pred)

for i in range(n_pred):
    frame_pred = pred_frames[i]
    label_pred = pred_labels[i]
    # print(frame_pred)
    # print(label_pred)
    iou_max = -1
    gt_match = -1

    for j in range(n_gt):
        if (gt_frames[j] != frame_pred):
            continue

        label_gt = gt_labels[j]
        iou = np.sum(label_pred * label_gt) / np.sum((label_pred + label_gt) > 0)

        if iou > iou_max:
            iou_max = iou
            gt_match = j

    if iou_max >= 0.5:
        if gt_match >= 0:
            if not np.any(tp[0:i] == gt_match):
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    else:
        fp[i] = 1
# for i in range(tp.size):
#     print(tp[i])

tp_cumsum = np.cumsum(tp)
fp_cumsum = np.cumsum(fp)
recall = tp_cumsum / n_gt
precision = tp_cumsum / (tp_cumsum + fp_cumsum)

# compute average precision
ap = 0
for t in np.arange(0, 1.1, 0.1):
    mask = recall >= t
    if np.any(mask):
        p = np.max(precision[mask])
    else:
        p = 0
    ap = ap + p / 11

print("Average Precision:", ap)
