import numpy as np
import cv2
import matplotlib.pyplot as plt

from core.config import cfg
import core.test
from core.test import im_detect_bbox
from core.test_engine import initialize_model_from_cfg, get_roidb_and_dataset
import utils.boxes as box_utils

from datasets.json_dataset import JsonDataset
from datasets import json_dataset_evaluator

from pycocotools.cocoeval import COCOeval
from utils.io import save_object

import logging
logger = logging.getLogger(__name__)


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def evaluate_box_proposals(
    roidb, proposals, thresholds=None, area='all', limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        'all': 0,
        'small': 1,
        'medium': 2,
        'large': 3,
        '96-128': 4,
        '128-256': 5,
        '256-512': 6,
        '512-inf': 7}
    area_ranges = [
        [0**2, 1e5**2],    # all
        [0**2, 32**2],     # small
        [32**2, 96**2],    # medium
        [96**2, 1e5**2],   # large
        [96**2, 128**2],   # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2]]  # 512-inf
    assert area in areas, 'Unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    # matched_box_overlaps = np.zeros(0)
    matched_box_overlaps = []
    matched_box_confidence = []
    matched_box_ind = []
    matched_gt_ind = []  # due to area filter, indices may shift
    num_pos = 0
    for i, entry in enumerate(roidb):
        assert (entry['gt_classes'] > 0).all()
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        # assert len(gt_inds) == len(entry['boxes']),\
        #     "{} vs {}".format(len(gt_inds), len(entry['boxes']))
        gt_boxes = entry['boxes'][gt_inds, :]
        gt_areas = entry['seg_areas'][gt_inds]
        valid_gt_inds = np.where(
            (gt_areas >= area_range[0]) & (gt_areas <= area_range[1]))[0]
        gt_boxes = gt_boxes[valid_gt_inds, :]  # note
        num_pos += len(valid_gt_inds)
        boxes = proposals['boxes'][i]
        box_confidence = proposals['scores'][i]
        boxes = box_utils.clip_boxes_to_image(
            boxes, entry['height'], entry['width']
        )
        if boxes.shape[0] == 0:
            continue
        if limit is not None and boxes.shape[0] > limit:
            boxes = boxes[:limit, :]
        overlaps = box_utils.bbox_overlaps(
            boxes.astype(dtype=np.float32, copy=False),
            gt_boxes.astype(dtype=np.float32, copy=False))
        _matched_box_overlaps = np.zeros((gt_boxes.shape[0]))
        _matched_box_confidence = np.zeros((gt_boxes.shape[0]))
        _matched_box_ind = -1 * np.ones((gt_boxes.shape[0]), dtype=np.int32)
        _matched_gt_ind = -1 * np.ones((gt_boxes.shape[0]), dtype=np.int32)
        for j in range(min(boxes.shape[0], gt_boxes.shape[0])):
            # find which proposal box maximally covers each gt box
            argmax_overlaps = overlaps.argmax(axis=0)
            # and get the iou amount of coverage for each gt box
            max_overlaps = overlaps.max(axis=0)
            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ind = max_overlaps.argmax()
            gt_ovr = max_overlaps.max()
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _matched_box_overlaps[j] = overlaps[box_ind, gt_ind]
            _matched_box_confidence[j] = box_confidence[box_ind]
            _matched_box_ind[j] = box_ind
            _matched_gt_ind[j] = gt_inds[valid_gt_inds][gt_ind]
            assert _matched_box_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
        # append recorded iou coverage level
        assert (_matched_gt_ind >= 0).all()
        matched_box_overlaps.append(_matched_box_overlaps)
        matched_box_confidence.append(_matched_box_confidence)
        matched_box_ind.append(_matched_box_ind)
        matched_gt_ind.append(_matched_gt_ind)

    gt_overlaps = np.sort(np.concatenate(matched_box_overlaps))
    if thresholds is None:
        step = 0.05
        thresholds = np.arange(0.5, 0.95 + 1e-5, step)
    recalls = np.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    ar = recalls.mean()
    return {
        'matched_box_overlaps': matched_box_overlaps,
        'matched_box_confidence': matched_box_confidence,
        'matched_box_ind': matched_box_ind,
        'matched_gt_ind': matched_gt_ind,
        'recalls': recalls,
        'ar': ar,
        'thresholds': thresholds,
        'num_pos': num_pos,
    }


def plot_gt_box_pair_over_img(im_fname, gt_box, matched_box, dpi=200):
    im = cv2.imread(im_fname)[:, :, ::-1]

    def _add_box(ax, bbox, color):
        print("adding box with area {:.2f}^2".format(
            np.sqrt( (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) )
        ))
        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False, edgecolor=color,
                linewidth=2
            )
        )

    fig, ax = plt.subplots(1)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax.imshow(im)
    _add_box(ax, gt_box, 'yellow')
    _add_box(ax, matched_box, 'r')
    plt.show()


def plot_gt_vs_proposal(dataset, roidbs, proposals, eval_res, inx_tup, dpi=60):
    img_inx, gt_inx = inx_tup
    print(inx_tup)
    print('overlap is {:.4f}'.format(
        eval_res['matched_box_overlaps'][img_inx][gt_inx] ))
    roidb = roidbs[img_inx]
    im_fname = roidb['image']
    # due to area filtering and others, gt indices may shift
    rio_gt_inx = eval_res['matched_gt_ind'][img_inx][gt_inx]
    matched_box_inx = eval_res['matched_box_ind'][img_inx][gt_inx]
    box_confidence = eval_res['matched_box_confidence'][img_inx][gt_inx]

    gt_cls = roidb['gt_classes'][rio_gt_inx]
    gt_box = roidb['boxes'][rio_gt_inx]
    cls_name = dataset.classes[gt_cls]
    print("img id: {}, gt class: {}, confidence: {:.3f}".format(
        roidb['id'], cls_name, box_confidence))

    if matched_box_inx == -1:
        print('no matching box')
        matched_box = (0, 0, 0, 0)
        raise ValueError("what")
    else:
        matched_box = proposals['boxes'][img_inx][matched_box_inx]
    plot_gt_box_pair_over_img(
        im_fname, gt_box, matched_box, dpi
    )


def ordered_plot_gt_vs_proposal(
        dataset, roidbs, proposals, eval_res, order, inx, dpi=60):
    if 'segment_double_inx' not in eval_res:
        segment_double_inx = []
        for i, segment in enumerate(eval_res['matched_box_overlaps']):
            size = len(segment)
            segment_double_inx.append(
                np.vstack([
                    i * np.ones(size, dtype=np.int32),
                    np.arange(size, dtype=np.int32)
                ])
            )
        eval_res['segment_double_inx'] = np.concatenate(segment_double_inx, axis=1)
    segment_double_inx = eval_res['segment_double_inx']
    inx_tup = tuple(segment_double_inx[:, order][:, inx])
    plot_gt_vs_proposal(dataset, roidbs, proposals, eval_res, inx_tup, dpi)



def collect_bbox_pred(args, dataset_name, proposal_file):
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'
    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, None
    )
    model = initialize_model_from_cfg(args)
    scores = []
    boxes = []
    for i, entry in enumerate(roidb):
        # if i > 100:
        #     break
        logger.info("currently processing image {}".format(i))
        if cfg.TEST.PRECOMPUTED_PROPOSALS:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select only the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = entry['boxes'][entry['gt_classes'] == 0]
            if len(box_proposals) == 0:
                continue
        else:
            # Faster R-CNN type models generate proposals on-the-fly with an
            # in-network RPN; 1-stage models don't require proposals.
            box_proposals = None
            if not cfg.RPN.RPN_ON:  # means we are doing gt testing. LOL
                box_proposals = entry['boxes']
                if len(box_proposals) == 0:
                    continue
        im = cv2.imread(entry['image'])
        _scores, _boxes, im_scale, _ = im_detect_bbox(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, box_proposals)
        scores.append(_scores)
        boxes.append(_boxes)
    return {
        'scores': scores,
        'boxes': boxes
    }


def box_results_with_nms_and_limit(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = 81
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > 0.05)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        keep = box_utils.nms(dets_j, 0.5)
        nms_dets = dets_j[keep, :]
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    DETECTIONS_PER_IM = 100
    image_scores = np.hstack(
        [cls_boxes[j][:, -1] for j in range(1, num_classes)]
    )
    if len(image_scores) > DETECTIONS_PER_IM:
        image_thresh = np.sort(image_scores)[-DETECTIONS_PER_IM]
        for j in range(1, num_classes):
            keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
            cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def mAP_eval(res):
    cls_boxes = [
        box_results_with_nms_and_limit(
            res['prob'][i], res['pred_box'][i]
        )[-1]
        for i in range(5000)
    ]
    cls_boxes = list(zip(*cls_boxes))  # transposed
    dset = JsonDataset('coco_2017_val')
    coco_eval = json_dataset_evaluator.evaluate_boxes(
        dset, cls_boxes, output_dir='/scratch/', cleanup=True)
    return coco_eval


def evaluate_json_bbox_result(res_file, output_fname=None):
    json_dataset = JsonDataset('coco_2017_val')
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # eval_file = os.path.join('./', 'old_results.pkl')
    if output_fname:
        save_object(coco_eval, output_fname)
        logger.info('Wrote json eval results to: {}'.format(output_fname))
    return coco_eval
