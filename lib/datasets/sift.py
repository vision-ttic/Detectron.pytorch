import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


def box_iou(dts, gts):
    assert len(dts) > 0
    assert len(gts) > 0
    dt_boxes = [ dt['bbox'] for dt in dts ]
    gt_boxes = [ gt['bbox'] for gt in gts ]
    iscrowd = np.array([ int(gt['iscrowd']) for gt in gts ])
    crowd_inx = np.where(iscrowd == 1)[0]
    ious = maskUtils.iou(dt_boxes, gt_boxes, iscrowd)
    ious[:, crowd_inx] = 0  # why this line at all?
    return ious


def img_category_agnostic_filter(cocoGt, cocoDt, imgId):
    """
    Still sort it by confidence
    """
    gts = cocoGt.imgToAnns[imgId]
    dts = cocoDt.imgToAnns[imgId]
    if len(dts) == 0 or len(gts) == 0:
        return
    dtScores = np.array([ dt['score'] for dt in dts ])
    order = np.argsort(-1 * dtScores)  # desc
    dts = [ dts[i] for i in order ]
    dtScores = [ dtScores[i] for i in order ]

    # hard_fp as well as best match of each object
    dts_admitted = []

    ious = box_iou(dts, gts)
    # hard_fp_mask = ious.sum(axis=1) == 0
    # those which matched no object of any category
    hard_fp_inx = np.where(ious.sum(axis=1) == 0)[0]
    hard_fp_dts = [ dts[i] for i in hard_fp_inx ]
    dts_admitted.extend(hard_fp_dts)

    remaining_dts = [ dts[i] for i in range(len(dts)) if i not in hard_fp_inx ]
    if len(remaining_dts) == 0:
        cocoDt.imgToAnns[imgId] = dts_admitted
        return

    ious = box_iou(remaining_dts, gts)  # [DxG]
    gt_ownership = np.zeros_like(ious)
    owned_by_which_gt = ious.argmax(axis=1)
    assert owned_by_which_gt.shape[0] == len(remaining_dts) == len(ious)
    gt_ownership[range(len(owned_by_which_gt)), owned_by_which_gt] = 1

    # those gts that own none of the detections are discarded so as to apply
    # the argmax trick. Argmax returns 1st item if a column is all 0, which is
    # not what we want
    gt_ownership = gt_ownership[:, gt_ownership.sum(axis=0) > 0]
    assert gt_ownership.shape[1] == len(np.unique(owned_by_which_gt))

    first_non_zero_per_col, gt_ownership = first_per_col(gt_ownership)
    conf = [ remaining_dts[i]['score'] for i in first_non_zero_per_col ]
    for i in first_non_zero_per_col:
        remaining_dts[i]['score'] += 1.01

    dts_admitted.extend( [ remaining_dts[i] for i in first_non_zero_per_col ] )

    # first_non_zero_per_col, gt_ownership = first_per_col(gt_ownership)
    # dts_admitted.extend( [ remaining_dts[i] for i in first_non_zero_per_col ] )
    #
    # first_non_zero_per_col, gt_ownership = first_per_col(gt_ownership)
    # dts_admitted.extend( [ remaining_dts[i] for i in first_non_zero_per_col ] )

    # cocoDt.imgToAnns[imgId] = dts_admitted


def first_per_col(matrix):
    matrix = matrix[:, matrix.sum(axis=0) > 0]  # empty col ignored
    # trick: argmax gets the first of equals
    # this is the most confident dt (class agnostic) belonging to the gt group
    first_non_zero_per_col = np.argmax(matrix > 0, axis=0)
    matrix[ first_non_zero_per_col, range(matrix.shape[1]) ] = 0
    return np.unique(first_non_zero_per_col), matrix


def filter_dts_on_box_identity(coco_eval):
    cocoGt = coco_eval.cocoGt
    cocoDt = coco_eval.cocoDt
    imgIds = cocoDt.getImgIds()
    for id in imgIds:
        img_category_agnostic_filter(cocoGt, cocoDt, id)
    return cocoGt, cocoDt


def do_eval(coco_gt, coco_dt):
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval

# def do_eval(coco_eval):
#     coco_gt = coco_eval.cocoGt
#     coco_dt = coco_eval.cocoDt
#     coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()
#     return coco_eval


def change_and_see(coco_eval):
    cocoGt, cocoDt = filter_dts_on_box_identity(coco_eval)
    new_eval = do_eval(cocoGt, cocoDt)
    return new_eval


def get_average_pred_per_img(coco_dt):
    acc = []
    for k, dt_list in coco_dt.imgToAnns.items():
        if len(dt_list) == 0:
            print(k)
        assert isinstance(dt_list, list)
        acc.append( len(dt_list) )
    # assert len(acc) == 5000, "actual length: {}".format(len(acc))
    return np.mean(acc)


def plot_per_class_AP(coco_eval):
    prec = coco_eval.eval['precision']
    AP = [ prec[0, :, cls, 0, -1].mean() for cls in range(80) ]
    mAP = np.mean(AP)
    plt.plot(AP)
    plt.title("mAP: {}".format(mAP))


def _test():
    from fabric.utils.io import load_object
    import os.path as osp
    root = '/home-nfs/whc/lab/scale/exp/detection/runs/collect_res/'
    fname = osp.join(root, '006/output/coco_eval.pkl')
    cocoeavl_obj = load_object(fname)
    sifted = change_and_see(cocoeavl_obj)


if __name__ == '__main__':
    _test()
