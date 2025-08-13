import numpy as np


def compute_per_class_IoU(eval_segm, gt_segm, ignore=[]):

    check_size(eval_segm, gt_segm)

    union_cl, num_union_cl = union_classes(eval_segm, gt_segm)

    eval_mask, gt_mask = extract_both_masks(
        eval_segm, gt_segm, union_cl, num_union_cl)

    IoU = list([0]) * num_union_cl

    for i, cl in enumerate(union_cl):
        if cl in ignore:
            continue

        eval_mask_cl = eval_mask[i, :, :]
        gt_mask_cl = gt_mask[i, :, :]

        intersection = np.logical_and(eval_mask_cl, gt_mask_cl)
        union = np.logical_or(eval_mask_cl, gt_mask_cl)
        IoU[i] = np.sum(intersection) / np.sum(union)

    return IoU, union_cl


def compute_FmIoU(eval_segm, gt_segm, ignore=[]):
    """
    Calculate F-mIoU (Frequency Weighted Intersection over Union)
    :param eval_segm: Model output segmentation result
    :param gt_segm: Ground truth segmentation
    :param ignore: List of classes to ignore
    :return: F-mIoU value
    """
    # Calculate IoU and class index for each class
    IoU, union_cl = compute_per_class_IoU(eval_segm, gt_segm, ignore)

    # Calculate class frequency
    total_pixels = gt_segm.size
    class_frequencies = []
    for cl in union_cl:
        if cl in ignore:
            class_frequencies.append(0)
        else:
            class_frequency = np.sum(gt_segm == cl) / total_pixels
            class_frequencies.append(class_frequency)

    # Calculate F-mIoU
    FmIoU = 0
    for i, cl in enumerate(union_cl):
        if cl in ignore:
            continue
        FmIoU += class_frequencies[i] * IoU[i]

    return FmIoU


def compute_per_class_accuracy(eval_segm, gt_segm, ignore=[]):
    """
    Calculate accuracy for each class
    :param eval_segm: Model output segmentation result
    :param gt_segm: Ground truth segmentation
    :param ignore: List of classes to ignore
    :return: Accuracy list for each class and class index list
    """
    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracies = list([0]) * n_cl

    for i, c in enumerate(cl):
        if c in ignore:
            continue

        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        true_positive = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        total_gt = np.sum(curr_gt_mask)

        if total_gt == 0:
            accuracies[i] = 0
        else:
            accuracies[i] = true_positive / total_gt

    return accuracies, cl


def get_ignore_classes_num(cl, ignore):
    """
    Returns the number of classes to ignore
    :param cl: list of classes
    :param ignore: list of classes to ignore
    :return: number of classes to ignore
    """
    overlap = [c for c in cl if c in ignore]
    return overlap, len(overlap)


def compute_mAcc(eval_segm, gt_segm, ignore=[]):
    """
    Calculate mean Accuracy (mAcc)
    :param eval_segm: Model output segmentation result
    :param gt_segm: Ground truth segmentation
    :param ignore: List of classes to ignore
    :return: Mean Accuracy
    """
    per_class_accuracies, _ = compute_per_class_accuracy(
        eval_segm, gt_segm, ignore)

    # Filter out classes to ignore
    valid_accuracies = [acc for acc in per_class_accuracies if acc is not None]

    if len(valid_accuracies) == 0:
        mAcc = 0
    else:
        mAcc = np.mean(valid_accuracies)

    return mAcc


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    """"
    Extracts the masks of the segmentation
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :param cl: list of classes
    :param n_cl: number of classes
    :return: masks of the segmentation
    """
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_masks(segm, cl, n_cl):
    """
    Extracts the masks of the segmentation
    :param segm: 2D array, segmentation
    :param cl: list of classes
    :param n_cl: number of classes
    :return: masks of the segmentation
    """
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def union_classes(eval_segm, gt_segm):
    """
    Returns the union of the classes
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    :return: union of the classes
    """
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)

    n_cl = len(cl)

    return cl, n_cl


def extract_classes(segm):
    """
    Extracts the classes from the segmentation
    :param segm: 2D array, segmentation
    :return: classes and number of classes
    """
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def segm_size(segm):
    """
    Returns the size of the segmentation
    :param segm: 2D array, segmentation
    :return: size of the segmentation
    """
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    """
    Checks the size of the segmentation
    :param eval_segm: 2D array, predicted segmentation
    :param gt_segm: 2D array, ground truth segmentation
    """
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


def get_pixel_area(segm):
    """
    Returns the area of the segmentation
    :param segm: 2D array, segmentation
    :return: area of the segmentation
    """
    return segm.shape[0] * segm.shape[1]


"""
Exceptions
"""


class EvalSegErr(Exception):
    """
    Custom exception for errors during evaluation
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
