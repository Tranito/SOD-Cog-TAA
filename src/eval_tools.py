import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline

def evaluate_earliness(all_pred, all_labels, time_of_accidents, fps=30.0, thresh=0.5):
    """Evaluate the earliness for true positive videos"""
    time = 0.0
    counter = 0
    # iterate each video sample
    for i in range(len(all_pred)):
        pred_bins = (all_pred[i] >= thresh).astype(int)
        inds_pos = np.where(pred_bins > 0)[0]
        if all_labels[i] > 0 and len(inds_pos) > 0:
            # only true positive video needs to compute earliness
            time += max((time_of_accidents[i] - inds_pos[0]) / fps, 0)
            counter += 1  # number of TP videos
    mTTA = time / counter if counter > 0 else 0 # average TTA (seconds) per-video
    return mTTA
def evaluation(all_pred, all_labels, time_of_accidents, fps=30.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    n_frames = 0

    #Iterate through each video and its corresponding time of accident (time_of_accidents).
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            #For positive videos (all_labels[idx] > 0), consider predictions up to the time of the accident.
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            #For negative videos, consider all predictions.
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        #Update min_pred with the minimum prediction value found in the current video 
        # if minimum prediciton is less then minimum prediction value found in current video 
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        #Add prediction to list of predictions
        preds_eval.append(pred)
        #Increase number of frames based on number of predictions in the current video
        n_frames += len(pred)

    #Determine total seconds based on number of frames and framer per second
    total_seconds = all_pred.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.1):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            #For each video, determine the frames where the prediction exceeds the threshold and the ground truth label is positive (tp).
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            #Update Tp if there is at least one true positive frame.
            Tp += float(len(tp[0])>0)
            # print(f"Value of Tp {Tp}")
            
            #If there is a least one TP, determine relative Time-to-Accident 
            # which is the ratio between the first TP and time of accident frame
            
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            
            #In DADA-2000 if video label is 1 then all frames have label 1
            #This implies that all predictions above thresholds are TP

            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
            # print(f"Tp_Fp: {Tp_Fp}")

        #Using total number of counted TP frames and all positive frames, 
        # determine precision and recall and TTA for each threshold

        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
        
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]

    #Sort the precision, recall, and time metrics by recall in ascending order.

    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    return AP, mTTA, TTA_R80


def print_results(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, Unc_all, result_dir):
    result_file = os.path.join(result_dir, 'eval_all.txt')
    with open(result_file, 'w') as f:
        for e, APvid, AP, mTTA, TTA_R80, Un in zip(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, Unc_all):
            f.writelines('Epoch: %s,'%(e) + ' APvid={:.3f}, AP={:.3f}, mTTA={:.3f}, TTA_R80={:.3f}, mAU={:.5f}, mEU={:.5f}\n'.format(APvid, AP, mTTA, TTA_R80, Un[0], Un[1]))
    f.close()


def vis_results(vis_data, batch_size, vis_dir, smooth=False, vis_batchnum=2):
    assert vis_batchnum <= len(vis_data)
    #Iterate over specified number of batches
    for b in range(vis_batchnum):
        #For each batch, it extracts the relevant data: predicted frames (pred_frames), labels (labels), 
        # time of accident (toa), video IDs (video_ids), detections (detections), and uncertainties (uncertainties).
        results = vis_data[b]
        pred_frames = results['pred_frames']
        labels = results['label']
        toa = results['toa']
        video_ids = results['video_ids']
        # detections = results['detections']
        # uncertainties = results['pred_uncertain']

        #Loop Over Samples in Batch
        for n in range(batch_size):
            #Extract predicted mean, aleatoric uncertainty and epistemic uncertainty for current sample
            # print(f"Value of n: {n}")
            pred_mean = pred_frames[n][:]  # (90,)

            # pred_std_alea = 1.0 * np.sqrt(uncertainties[n, :, 0])
            # pred_std_epis = 1.0 * np.sqrt(uncertainties[n, :, 1])

            #xvals is initialized as a range object representing the frame indices.
            xvals = range(len(pred_mean))
            #If smooth is True, the function performs smoothing on the predictions.
            #It reduces the number of points in pred_mean, pred_std_alea, and pred_std_epis using linear interpolation.
            if smooth:
                # sampling
                xvals = np.linspace(0,len(pred_mean)-1,20)
                pred_mean_reduce = pred_mean[xvals.astype(int)]

                # pred_std_alea_reduce = pred_std_alea[xvals.astype(np.int)]
                # pred_std_epis_reduce = pred_std_epis[xvals.astype(np.int)]

                # smoothing
                #It then smooths the reduced points using cubic spline interpolation.
                xvals_new = np.linspace(1,len(pred_mean)+1,80)
                pred_mean = make_interp_spline(xvals, pred_mean_reduce)(xvals_new)

                # pred_std_alea = make_interp_spline(xvals, pred_std_alea_reduce)(xvals_new)
                # pred_std_epis = make_interp_spline(xvals, pred_std_epis_reduce)(xvals_new)

                #The smoothed predictions are clipped to a maximum value of 1.0 - 1e-3.
                pred_mean[pred_mean >= 1.0] = 1.0-1e-3
                xvals = xvals_new
                # fix invalid values
                # indices = np.where(xvals <= toa[n])[0]
                # xvals = xvals[indices]
                # pred_mean = pred_mean[indices]

                # pred_std_alea = pred_std_alea[indices]
                # pred_std_epis = pred_std_epis[indices]

            # plot the probability predictions
            fig, ax = plt.subplots(1, figsize=(24, 3.5))
            #The fill_between function is used to create shaded regions around the predicted mean (pred_mean).
            #The region between pred_mean - pred_std_alea and pred_mean + pred_std_alea is shaded in wheat color to represent aleatoric uncertainty.

            # ax.fill_between(xvals, pred_mean - pred_std_alea, pred_mean + pred_std_alea, facecolor='wheat', alpha=0.5)

            #The region between pred_mean - pred_std_epis and pred_mean + pred_std_epis is shaded in yellow color to represent epistemic uncertainty.

            # ax.fill_between(xvals, pred_mean - pred_std_epis, pred_mean + pred_std_epis, facecolor='yellow', alpha=0.5)

            plt.plot(xvals, pred_mean, linewidth=3.0)
            #Add vertical dashed line that represents the time of accident frame if it is within the frame range
            if toa[n] <= 150:
                plt.axvline(x=toa[n], ymax=1.0, linewidth=3.0, color='r', linestyle='--')
            #Add text to indicate that the vertical dashed line represents the time of accident frame
            plt.text(toa[n] + 1, 0, "Accident", fontsize = 22, color = "r")

            #Add horizontal dashed line that represents the threshold value of 0.5
            plt.axhline(y=0.5, xmin=0, xmax=150, linewidth=3.0, color='g', linestyle='--')
            #Add text to indicate that the horizontal dashed line represents the threshold value of 0.5
            plt.text( 0.5, 0.55, "Threshold", fontsize = 18, color = "g")
            
            
            # draw accident region
            x = [toa[n], pred_frames.shape[1]]
            y1 = [0, 0]
            y2 = [1, 1]
            ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
            fontsize = 25
            plt.ylim(0, 1.1)
            plt.xlim(1, pred_frames.shape[1])
            plt.ylabel('Probability', fontsize=fontsize)
            plt.xlabel('Frame (FPS=30)', fontsize=fontsize)
            plt.xticks(range(0, pred_frames.shape[1] + 10, 10), fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.grid(True)
            plt.tight_layout()
            tag = 'pos' if labels[n] > 0 else 'neg'
            #filename: sample_id (from testing.txt file) + video_id + tag + '.png'
            plt.savefig(os.path.join(vis_dir, f"{video_ids[n][0]}" + "_" + video_ids[n][1] + '_' + tag + '.png'))
            plt.close()
            # plt.show()