import numpy as np
import cv2

scale_subregion = float(3) / 4
scale_mask = float(1)/(scale_subregion*4)
def calculate_iou(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j)/float(i))
    return iou

def calculate_gtc(bbox, gtbbox):
    gt_area = (gtbbox[2] - gtbbox[0]) * (gtbbox[3] - gtbbox[1])
    
    intersect_width = max(min(gtbbox[2], bbox[2]) - max(gtbbox[0], bbox[0]), 0)
    intersect_height = max(min(gtbbox[3], bbox[3]) - max(gtbbox[1], bbox[1]), 0)
    intersect_area = intersect_width * intersect_height
    gtc = intersect_area / gt_area
    return gtc

def calculate_cd(bbox, gtbbox, image_shape):
    #image_shape = (281, 500)
    diagonal_length = np.linalg.norm(np.array(image_shape))
    gt_center = np.array([(gtbbox[1] + gtbbox[3]) / 2, (gtbbox[0] + gtbbox[2]) / 2])
    pred_center = np.array([(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2])
    cd = np.linalg.norm(gt_center - pred_center)
    normalized_cd = cd / diagonal_length
    return normalized_cd

def calculate_overlapping(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(gt_mask)
    overlap = float(float(j)/float(i))
    return overlap


def follow_iou(gt_masks, region_mask, classes_gt_objects, class_object, last_matrix):
    results = np.zeros([np.size(array_classes_gt_objects), 1])
    for k in range(np.size(classes_gt_objects)):
        if classes_gt_objects[k] == class_object:
            gt_mask = gt_masks[:, :, k]
            iou = calculate_iou(region_mask, gt_mask)
            results[k] = iou
    index = np.argmax(results)
    new_iou = results[index]
    iou = last_matrix[index]
    return iou, new_iou, results, index

# Auto find the max bounding box in the region image
def find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, class_object, improve_reward, original_shape):
    _, _, n = gt_masks.shape
    max_iou = 0.0
    max_metr = 0.0
    for k in range(n):
        if classes_gt_objects[k] != class_object:
            continue
        gt_mask = gt_masks[:,:,k]
        iou = calculate_iou(region_mask, gt_mask)
        #region_box = get_bounding_box_from_mask(region_mask)
        #gt_box = get_bounding_box_from_mask(gt_mask)
        region_box = get_bounding_box_from_masks(region_mask)
        gt_box = get_bounding_box_from_masks(gt_mask)
        #test_iou = intersection_over_union(region_box, gt_box)
        gtc = calculate_gtc(region_box, gt_box)
        cd = calculate_cd(region_box, gt_box, original_shape)
        metr = iou + gtc + 1 - cd
        if improve_reward:
            if max_metr < metr:
                max_iou = iou
                max_metr = metr
        else:
            if max_iou < iou:
                max_iou = iou
                max_metr = metr
    return max_iou, max_metr

def find_max_bounding_box_origin(gt_masks, region_mask, classes_gt_objects, class_object):
    _, _, n = gt_masks.shape
    max_iou = 0.0
    for k in range(n):
        if classes_gt_objects[k] != class_object:
            continue
        gt_mask = gt_masks[:,:,k]
        iou = calculate_iou(region_mask, gt_mask)
        if max_iou < iou:
            max_iou = iou
    return max_iou

def get_crop_image_and_mask(original_shape, offset, region_image, size_mask, action):
    r"""crop the the image according to action
    
    Args:
        original_shape: shape of original image (H x W)
        offset: the current image's left-top coordinate base on the original image
        region_image: the image to be cropped
        size_mask: the size of region_image
        action: the action choose by agent. can be 1,2,3,4,5.
        
    Returns:
        offset: the cropped image's left-top coordinate base on original image
        region_image: the cropped image
        size_mask: the size of the cropped image
        region_mask: the masked image which mask cropped region and has same size with original image
    
    """
    
    
    region_mask = np.zeros(original_shape) # mask at original image 
    size_mask = (int(size_mask[0] * scale_subregion), int(size_mask[1] * scale_subregion)) # the size of croped image
    if action == 1:
        offset_aux = (0, 0)
    elif action == 2:
        offset_aux = (0, int(size_mask[1] * scale_mask))
        offset = (offset[0], offset[1] + int(size_mask[1] * scale_mask))
    elif action == 3:
        offset_aux = (int(size_mask[0] * scale_mask), 0)
        offset = (offset[0] + int(size_mask[0] * scale_mask), offset[1])
    elif action == 4:
        offset_aux = (int(size_mask[0] * scale_mask), 
                      int(size_mask[1] * scale_mask))
        offset = (offset[0] + int(size_mask[0] * scale_mask),
                  offset[1] + int(size_mask[1] * scale_mask))
    elif action == 5:
        offset_aux = (int(size_mask[0] * scale_mask / 2),
                      int(size_mask[0] * scale_mask / 2))
        offset = (offset[0] + int(size_mask[0] * scale_mask / 2),
                  offset[1] + int(size_mask[0] * scale_mask / 2))
    region_image = region_image[offset_aux[0]:offset_aux[0] + size_mask[0],
                   offset_aux[1]:offset_aux[1] + size_mask[1]]
    region_mask[offset[0]:offset[0] + size_mask[0], offset[1]:offset[1] + size_mask[1]] = 1
    return offset, region_image, size_mask, region_mask

def get_bounding_box_from_mask(mask):
    print(mask)
    print(mask.shape)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return [x, y, x + w, y + h]  # [xmin, ymin, xmax, ymax]
    else:
        return [0, 0, 0, 0]
    
def intersection_over_union(box1, box2):
        """
            Calcul de la mesure d'intersection/union
            Entrée :
                Coordonnées [x_min, x_max, y_min, y_max] de la boite englobante de la vérité terrain et de la prédiction
            Sortie :
                Score d'intersection/union.

        """
        x11, y11, x21, y21 = box1
        
        x12, y12, x22, y22 = box2
        
        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

def get_bounding_box_from_masks(masks):
    # 找到非零值的索引
    nonzero_indices = np.nonzero(masks)
    
    # 如果有非零值
    if len(nonzero_indices[0]) > 0:
        # 獲得最小和最大的 x 和 y 座標
        xmin = np.min(nonzero_indices[1])
        ymin = np.min(nonzero_indices[0])
        xmax = np.max(nonzero_indices[1])
        ymax = np.max(nonzero_indices[0])
        
        return xmin, ymin, xmax, ymax
        
    else:
        return None, None, None, None