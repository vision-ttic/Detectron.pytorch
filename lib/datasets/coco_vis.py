import os.path as osp
from collections import defaultdict
import functools

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from pycocotools import mask as maskUtils

from ipywidgets import interactive  # , fixed, interact_manual
import ipywidgets as widgets

from IPython.display import display

from string import Template

from scipy.misc import imread


def read_coco_img(file_name, datadir='/share/data/vision-greg/coco/images/val2017'):
    abs_path = osp.join(datadir, str(file_name))
    im = cv2.imread(abs_path)
    # cv2 requires rbg channel switching. Need to investigate. It may have
    # something to do with detectron global setting. Be careful
    im = im[:, :, ::-1]
    return im


def read_voc_img(file_name, year):
    assert year in ['07', '12']
    data_dir_template = \
        Template("/share/data/vision-greg2/users/whc/lab/pdet/data/VOC20$Y/JPEGImages")
    abs_path = osp.join(
        data_dir_template.substitute(Y=year),
        str(file_name)
    )
    im = cv2.imread(abs_path)
    im = im[:, :, ::-1]
    return im


def read_voc_seg(file_name, year='07'):
    assert year in ['07', '12']
    data_dir_template = \
        Template("/share/data/vision-greg2/users/whc/lab/pdet/data/VOC20$Y/SegmentationClass")
    abs_path = osp.join(
        data_dir_template.substitute(Y=year),
        str(file_name)
    )
    abs_path = abs_path.replace('jpg', 'png')
    im = imread(abs_path)
    # im = im[:, :, ::-1]
    return im


def plot_image_with_anns(img, detections=(), gtruths=(), dpi=80):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
    ax.imshow(img)
    add_detections(detections)
    add_gtruths(gtruths, img)
    plt.show()


def add_detections(anns):
    ax = plt.gca()
    for ann in anns:
        if ann is None:
            continue
        bbox = ann['bbox']
        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                fill=False, edgecolor='orange', linewidth=3
            )
        )


def add_gtruths(anns, img):
    ax = plt.gca()
    # ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        if ann is None:
            continue
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        bbox = ann['bbox']
        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                fill=False, facecolor=None, edgecolor=c, linewidth=2
            )
        )

        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    polygons.append(Polygon(poly))
                    color.append(c)
            else:
                print("using crowd mask")
                height, width = img.shape[:-1]
                # mask
                if type(ann['segmentation']['counts']) == list:
                    rle = maskUtils.frPyObjects(
                        [ann['segmentation']], height, width
                    )
                else:
                    rle = [ann['segmentation']]
                m = maskUtils.decode(rle)
                img = np.ones( (m.shape[0], m.shape[1], 3) )
                if ann['iscrowd'] == 1:
                    color_mask = np.array([2.0, 166.0, 101.0]) / 255
                if ann['iscrowd'] == 0:
                    color_mask = np.random.random((1, 3)).tolist()[0]
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack( (img, m * 0.5) ))

    p = PatchCollection(
        polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


def logger(msg):
    # logging by wdgt about its levels
    print("------- {}".format(msg))


class Visualizer():
    def __init__(self, coco_eval):
        self.coco_eval = coco_eval
        # self.cats_name_to_json_id = cats_name_to_json_id
        self.cats_name_to_json_id = {
            '{}: {}'.format(meta['supercategory'], meta['name']): json_id
            for json_id, meta in coco_eval.cocoGt.cats.items()
        }
        self.jsonId_to_catsName = {
            jId: name
            for name, jId in self.cats_name_to_json_id.items()
        }
        # note that detectron json_dataset.py uses 1 based contId
        # for convenience I use 0 based.
        self.jsonId_to_contId = {
            v: i
            for i, v in enumerate(coco_eval.cocoGt.getCatIds())
        }
        self.contId_to_catsName = {
            contId: self.jsonId_to_catsName[jId]
            for jId, contId in self.jsonId_to_contId.items()
        }
        self.current_widget = None

        # global wdgts involve moderately intensive computations and are cached
        # they are deleted and swapped out on category change
        self.global_curr_imgId = None
        self.global_walk_matched_wdgt = None
        self.global_walk_missed_wdgt = None
        # ---- get the order for category display
        AP_over_cls = []
        for k, json_id in self.cats_name_to_json_id.items():
            cls = self.jsonId_to_contId[json_id]  # - 1
            ap = coco_eval.eval['scores'][0, :, cls, 0, -1].mean()
            AP_over_cls.append(ap)
        AP_cls_order = np.argsort(AP_over_cls)
        self.ordered_cat_name = [
            list(self.cats_name_to_json_id.keys())[i]
            for i in AP_cls_order
        ]
        self.global_cat_button = self.make_category_button()

    def make_category_button(self):
        """
        Simple helper that's frequently called.
        Later add logic to sort the category display order from
        worst performing class to the best
        """
        return widgets.Select(
            options=self.ordered_cat_name,
            description='category', rows=15
        )

    def category_PR_curve(self):
        def logic(cat_name):
            category = cat_name
            cats = self.cats_name_to_json_id
            coco_eval = self.coco_eval
            cls = self.jsonId_to_contId[cats[category]]  # - 1
            ap = coco_eval.eval['precision'][0, :, cls, 0, -1].mean()
            print('AP@50 is {:.3f}'.format(ap))
            fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, squeeze=True)
            fig.set_figheight(4)
            fig.set_figwidth(23)
            for area_ind, area_tag in enumerate(['all', 'small', 'medium', 'large']):
                area_ap = coco_eval.eval['precision'][0, :, cls, area_ind, -1].mean()
                axes[area_ind].plot(coco_eval.eval['scores'][0, :, cls, area_ind, -1], label='score')
                axes[area_ind].plot(coco_eval.eval['precision'][0, :, cls, area_ind, -1], label='precision')
                axes[area_ind].set_title("area {}, ap {:.3f}".format(area_tag, area_ap))
                axes[area_ind].set_xlabel("recall lvl")
                axes[area_ind].legend()
        wdgt = interactive(
            logic,
            cat_name=self.global_cat_button
        )
        wdgt.children[-1].layout.height = '300px'
        display(wdgt)

    def image_vis(self):
        controller = self.controller_widget()
        controller.children[-1].layout.height = '1000px'
        display(controller)

    def controller_widget(self):
        """
        """
        def logic(mode='Global'):
            if mode == 'Global':
                logger("INFO: global by design is in walk mode.")
                wdgt = self.global_widgets()
                display(wdgt)
            elif mode == 'Onto Single':
                logger("INFO: single support global style walk on the detections "
                       "of this image as well as bulk view of detections and ground truths")
                wdgt = self.single_img_widget()
                display(wdgt)
            else:
                raise ValueError("Invalid mode")
        interface = interactive(
            logic,
            mode=widgets.ToggleButtons(
                options=['Global', 'Onto Single'],
                description='mode :'
            )
        )
        return interface

    def single_img_widget(self):
        #---- wdgt logic
        def logic(image_id, cat_name, mode):
            logger("INFO: mark widget boundary")
            if image_id not in self.coco_eval.cocoGt.imgs:
                raise ValueError("{} not a valid id".format(image_id))
            catId = self.cats_name_to_json_id[cat_name]
            if mode == 'walk':
                wdgt_type = 'single_walk_matched'
            elif mode == 'bulk':
                wdgt_type = 'single_bulk_view'
            wdgt = self.widget_factory(wdgt_type, [image_id], catId)
            display(wdgt)

        #---- interface specific
        def get_relevant_category_names(img_id):
            areaInx = 0
            cats_of_interest = []
            for name, json_id in self.cats_name_to_json_id.items():
                val = self.coco_eval.evalImgsDict.get((img_id, json_id, areaInx), None)
                if val is not None:
                    cats_of_interest.append(name)
            return tuple(cats_of_interest)

        default_id = self.global_curr_imgId
        image_id_button = widgets.IntText(
            value=default_id,
            description='image id'
        )
        category_button = self.make_category_button()
        category_button.options = get_relevant_category_names(default_id)
        category_button.value = self.global_cat_button.value

        def update_cat_of_interest(*args):
            imgId = image_id_button.value
            cats_of_interest = get_relevant_category_names(imgId)
            category_button.options = cats_of_interest

        image_id_button.observe(update_cat_of_interest, 'value')

        interface = interactive(
            logic,
            image_id=image_id_button,
            cat_name=category_button,
            mode=widgets.ToggleButtons(
                value='bulk',
                options=['walk', 'bulk'],
                description='mode:'
            )
        )
        return interface

    def global_widgets(self):
        def logic(cat_name, mode='matched'):
            logger("INFO: mark widget boundary")
            self.global_cat = cat_name
            catId = self.cats_name_to_json_id[cat_name]
            if mode == 'dts walk (tp + fp)':
                if self.global_walk_matched_wdgt is None or\
                        self.global_walk_matched_wdgt.catId != catId:
                    self.global_walk_matched_wdgt = self.widget_factory(
                        'global_walk_matched', self.coco_eval.params.imgIds, catId
                    )
                display(self.global_walk_matched_wdgt)
            elif mode == 'missed gts walk':
                if self.global_walk_missed_wdgt is None or\
                        self.global_walk_missed_wdgt.catId != catId:
                    self.global_walk_missed_wdgt = self.widget_factory(
                        'global_walk_missed', self.coco_eval.params.imgIds, catId
                    )
                display(self.global_walk_missed_wdgt)
            else:
                raise ValueError("Invalid mode")

        interface = interactive(
            logic,
            mode=widgets.ToggleButtons(
                options=['dts walk (tp + fp)', 'missed gts walk'],
            ),
            cat_name=self.global_cat_button
        )
        return interface

    def widget_factory(self, widget_type, imgIds, catId, areaInx=0):
        """
        factory for bottom layer widgets
        global and single img walk matched share 1 widget
        global watch missed share 1 widget
        single img use 1 widget
        """
        coco_eval = self.coco_eval
        acc = []
        for imgId in imgIds:
            _val = coco_eval.evalImgsDict[ (imgId, catId, areaInx) ]
            if _val is not None:
                acc.append(_val)

        thr = 0
        if len(acc) == 0:
            raise ValueError('Neither gt nor detections found')
        # print("length of acc {}".format(len(acc)))

        # ----- ground truths
        gtm = np.concatenate([elem['gtMatches'] for elem in acc], axis=1)
        # gt_order = np.argsort(-gtm, axis=1) # unlikely to be useful
        gtIds = np.concatenate([elem['gtIds'] for elem in acc])
        gtIgs = np.concatenate([elem['gtIgnore'] for elem in acc]).astype(np.bool)
        total_valid = sum(~gtIgs)

        # used by global walk missed
        # this handles gracefully even in the absence of matched gts
        imgId_to_unmatched_gtId_dict = defaultdict(list)
        unmatched_gt_ids = gtIds[np.where(gtm[thr, :] == 0)]
        for _unmatched_gt_id in unmatched_gt_ids:
            gt = coco_eval.cocoGt.anns[_unmatched_gt_id]
            img_id = gt['image_id']
            imgId_to_unmatched_gtId_dict[img_id].append(gt)

        # ----- detections
        dtscores = np.concatenate([elem['dtScores'] for elem in acc])
        order = np.argsort(-dtscores, kind='mergesort')

        sorted_scores = dtscores[order]
        dtIds = np.concatenate([elem['dtIds'] for elem in acc], axis=0)[order]
        dtm = np.concatenate([elem['dtMatches'] for elem in acc], axis=1)[:,order]
        dtIgs = np.concatenate([elem['dtIgnore'] for elem in acc], axis=1)[:,order]

        tp = np.logical_and(               dtm,  np.logical_not(dtIgs))
        fp = np.logical_and(np.logical_not(dtm), np.logical_not(dtIgs))
        tp_sum = tp.cumsum(axis=1)
        fp_sum = fp.cumsum(axis=1)
        precision = tp_sum / (fp_sum + tp_sum + np.spacing(1))
        if total_valid > 0:
            recall = tp_sum / total_valid
        else:
            recall = None
            print("there is no valid gts")

        def walk_matched_logic(seq_inx):
            logger("INFO: mark widget boundary")
            confidence = sorted_scores[seq_inx]
            dt_id = dtIds[seq_inx]
            matched_gt_id = dtm[thr][seq_inx]
            is_ignored = dtIgs[thr][seq_inx]
            curr_prec = precision[thr][seq_inx]
            curr_rcll = recall[thr][seq_inx] if recall is not None else 0
            max_recall = recall[thr][-1] if recall is not None else 0
            print("walking at {}th dt out of {} dts".format(seq_inx, len(dtIds)))
            print("in total {} valid gts, {} recalled".format(
                total_valid, int(total_valid * max_recall)))
            print("")
            print(
                (
                    "dt id: {}, gt id: {}, dt forgiven: {}\n"
                    "prec so far: {:.3f}, rcll so far: {:.3f}, max rcll: {:.3f}\n"
                    "confidence: {:.3f}"
                ).format(dt_id, matched_gt_id, is_ignored,
                         curr_prec, curr_rcll, max_recall,
                         confidence)
            )
            dt = coco_eval.cocoDt.anns[dt_id]
            gt = coco_eval.cocoGt.anns.get(matched_gt_id, None)
            if gt is not None:
                assert gt['image_id'] == gt['image_id']
                iou = maskUtils.iou([dt['bbox']], [gt['bbox']], [False])
                print("box iou: {:.3f}".format(iou[0][0]))
            else:
                print("CURRENT DT UNMATCHED!")
            image_id = dt['image_id']
            print("img id: {}".format(image_id))

            im = read_coco_img(coco_eval.cocoDt.imgs[image_id]['file_name'])
            plot_image_with_anns(im, [dt], [gt])
            # WARN: global_curr_imgId is set even under single image mode.
            # because this routine is shared by both global and single walk wdgt
            # For now, logically there is no conflict since this affects only
            # the default image_id for single img wdgt. Can still manually change
            # Moreover, global walk updates this field as it browses; it's fresh
            self.global_curr_imgId = image_id

        def walk_matched_wdgt():
            if len(dtIds) == 0:
                raise ValueError("NO DETECTIONS IN THIS INFORMATION UNIT")

            # ------- incrementer start
            inx_incrementer = widgets.BoundedIntText(
                min=0, max=len(dtIds) - 1, step=1)
            # ------- incrementer end

            # ------- buttons start
            def on_click_handler(b):
                curr_seq_inx = inx_incrementer.value
                matched_mask = dtm[thr][:]
                if b.duty == 'fp':
                    inds = np.where(matched_mask == 0)[0]
                elif b.duty == 'tp':
                    inds = np.where(matched_mask > 0)[0]
                else:
                    raise ValueError("invalid button duty")
                if len(inds) == 0:
                    return
                # this causes a circle back behavior since argmax of all false gives 0
                if b.direction == 'next':
                    target = inds[np.argmax( inds > curr_seq_inx )]
                elif b.direction == 'prev':
                    offset = np.argmax( (inds < curr_seq_inx)[::-1] )
                    target = inds[ len(inds) - 1 - offset ]
                inx_incrementer.value = target

            button_acc = []
            for duty in ['fp', 'tp']:
                for direction in ['next', 'prev']:
                    desc = "{} {}".format(direction, duty)
                    button = widgets.Button(description=desc)
                    button.direction = direction
                    button.duty = duty
                    button.on_click(on_click_handler)
                    button_acc.append(button)
            # ------- buttons end

            def logic(seq_inx):
                # buttons cannot be inserted into interface. Have to be displayed
                # in the core logic itself
                ui = widgets.HBox(button_acc)
                display(ui)
                walk_matched_logic(seq_inx)

            interface = interactive(
                logic,
                seq_inx=inx_incrementer
            )
            interface.catId = catId
            return interface

        def global_walk_missed():
            def logic(seq_inx):
                logger("INFO: mark widget boundary")
                image_id = list(imgId_to_unmatched_gtId_dict.keys())[seq_inx]
                im = read_coco_img(coco_eval.cocoDt.imgs[image_id]['file_name'])
                unmatched_gts = imgId_to_unmatched_gtId_dict[image_id]
                print("total {} unmatched objects in {} images".format(
                    len(unmatched_gt_ids), len(imgId_to_unmatched_gtId_dict) ) )
                print("image id: {}".format(image_id))
                print("{} unmatched objects in this image".format(
                    len(unmatched_gts)) )
                plot_image_with_anns(im, detections=(), gtruths=unmatched_gts)
                self.global_curr_imgId = image_id
            interface = interactive(
                logic,
                seq_inx=widgets.BoundedIntText(
                    min=0, max=len(imgId_to_unmatched_gtId_dict) - 1, step=1)
            )
            interface.catId = catId
            return interface

        def single_bulk_view():
            def logic(which_to_display, dts_score_threshold):
                logger("INFO: mark widget boundary")
                assert len(imgIds) == 1
                image_id = imgIds[0]
                im = read_coco_img(coco_eval.cocoDt.imgs[image_id]['file_name'])

                def idsToAnns(split, ann_ids, score_threshold=None):
                    if split == 'gts':
                        src = self.coco_eval.cocoGt
                    elif split == 'dts':
                        acc = []  # this is super ugly
                        dtIds_list = list(dtIds)
                        for id in ann_ids:
                            index = dtIds_list.index(id)
                            if dtscores[index] >= dts_score_threshold:
                                acc.append(id)
                        ann_ids = acc
                        src = self.coco_eval.cocoDt
                    else:
                        raise ValueError("unrecognized split")
                    return [ src.anns[id] for id in ann_ids ]

                matched_gts = idsToAnns('gts', gtIds[np.where(gtm[thr, :] != 0)])
                unmatched_gts = idsToAnns('gts', gtIds[np.where(gtm[thr, :] == 0)])

                matched_dts = idsToAnns(
                    'dts', dtIds[np.where(dtm[thr, :] != 0)], dts_score_threshold)
                unmatched_dts = idsToAnns(
                    'dts', dtIds[np.where(dtm[thr, :] == 0)], dts_score_threshold)

                print( ("matched gt: {} unmatched gt: {}\n"
                        "matched_dt: {} unmatched dt: {}").format(
                    len(matched_gts), len(unmatched_gts),
                    len(matched_dts), len(unmatched_dts)
                ))

                gts_to_show = []
                dts_to_show = []
                for tranche in which_to_display:
                    if tranche.endswith('dts'):
                        dts_to_show += locals()[tranche]
                    elif tranche.endswith('gts'):
                        gts_to_show += locals()[tranche]
                plot_image_with_anns(
                    im, detections=dts_to_show, gtruths=gts_to_show)

            interface = interactive(
                logic,
                which_to_display=widgets.SelectMultiple(
                    options=[
                        'matched_gts', 'unmatched_gts',
                        'matched_dts', 'unmatched_dts'
                    ],
                    value=[
                        'matched_gts', 'unmatched_gts',
                        'matched_dts', 'unmatched_dts'
                    ],
                    rows=4,
                ),
                dts_score_threshold=widgets.FloatSlider(
                    value=0.50, min=0, max=1.0, step=0.05,
                    description='dt score threshold: ',
                    continuous_update=False,
                    readout=True, readout_format='.2f',
                )
            )
            return interface


        if widget_type == 'global_walk_matched':
            return walk_matched_wdgt()
        elif widget_type == 'global_walk_missed':
            return global_walk_missed()
        elif widget_type == 'single_walk_matched':
            return walk_matched_wdgt()
        elif widget_type == 'single_bulk_view':
            return single_bulk_view()
        else:
            raise NotImplementedError()
