import numpy as np

def ReadToolAnnotationFile(ground_truth_file):
    """
    This function reads the tool annotation file (ground truth).
    """
    with open(ground_truth_file, 'r') as f:
        # read the header first
        toolNames = f.readline().strip().split('\t')[1:]

        # read the labels
        gt = np.loadtxt(f, dtype=int, usecols=range(1, len(toolNames)+1))
        
    return gt, toolNames


def ReadToolPredictionFile(pred_file):
    """
    This function reads the tool prediction file (result).
    """
    with open(pred_file, 'r') as f:
        # read the header first
        f.readline()

        # read the labels
        pred = np.loadtxt(f, dtype=float)
        
    return pred


if __name__ == '__main__':
    ground_truth_files = [r'D:\SAP\CapStone Projects\data\train_dataset\tool_video_01.txt']

    # load all the data
    allGT = np.empty((0, len(ReadToolAnnotationFile(ground_truth_files[0])[0][0])))
    allPred = np.empty((0, len(ReadToolPredictionFile(ground_truth_files[0][:-4]+'_pred.txt')[0])))

    for ground_truth_file in ground_truth_files:
        pred_file = ground_truth_file[:-4]+'_pred.txt'

        gt, _ = ReadToolAnnotationFile(ground_truth_file)
        pred = ReadToolPredictionFile(pred_file)
        print(f"GT shape: {gt.shape}, pred shape: {pred.shape}")


        if gt.shape != pred.shape:
            raise ValueError(f"ERROR: {ground_truth_file}\nGround truth and prediction have different sizes")
        
        if not np.all(gt[:,0] == pred[:,0]):
            raise ValueError(f"ERROR: {ground_truth_file}\nThe frame index in ground truth and prediction is not equal")
        
        allGT = np.vstack([allGT, gt[:,1:]])
        allPred = np.vstack([allPred, pred])

    # compute average precision per tool
    ap = np.zeros(allGT.shape[1])
    allPrec = []
    allRec = []

    print('========================================')
    print('Average precision')
    print('========================================')

    for iTool in range(allGT.shape[1]):
        matScores = allPred[:, iTool]
        matGT = allGT[:, iTool]

        # sanity check, making sure it is confidence values
        X = np.unique(matScores)
        if len(X) == 2:
            print('- WARNING: the computation of mAP requires confidence values')
        
        # NEW Script - less sensitive to confidence ranges
        maxScore = np.max(matScores)
        minScore = np.min(matScores)
        step = (maxScore - minScore) / 2000
        
        if minScore == maxScore:
            raise ValueError('no difference confidence values')
        
        prec = []
        rec = []
        for iScore in np.arange(minScore, maxScore, step):
            bufScore = matScores > iScore
            tp = np.sum(np.double((bufScore == matGT) & (bufScore == 1)))
            fp = np.sum(np.double((bufScore != matGT) & (bufScore == 1)))

            if tp+fp != 0:
                rec.append(tp/np.sum(matGT > 0))
                prec.append(tp/(tp+fp))

        # compute average precision - finer way to compute AP
        ap.append(0)
        for t in np.arange(0, 1.1, 0.1):
            idx = np.where(abs(rec - t) < 0.00001)[0]
            p = np.mean(np.array(prec)[idx])

            if np.isnan(p):
                p = 0
            ap[-1] += p/11

        print(f"{toolNames[iTool]}: {ap[-1]}")

    print('----------------------------------------')
    print(f"All tools: {np.mean(ap)}")
    print('----------------------------------------')