from skimage.metrics import structural_similarity as compare_ssim
import numpy as np


class bb_utils():

  def bb_intersection_over_union(self, boxA, boxB):
    '''return the intersection over union value'''

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

  def associate_bounding_boxes(self, bbox_1, bbox_2):
      '''
      bbox_1: bounding box form image 1
      bbox_2: bounding box form image 2
      Define a Hungarian Matrix with IOU as a metric and return,
       for each box, an corresponding an id if match is found
      '''
      # Define a new IOU Matrix nxm with input boxes
      iou_matrix = np.zeros((len(bbox_1), len(bbox_2)), dtype=np.float32)

      # Go through boxes and store the IOU value for each box
      for i, box_1 in enumerate(bbox_1):
          for j, box_2 in enumerate(bbox_2):
              iou_matrix[i][j] = bb_intersection_over_union(box_1, box_2)

      # Call for the Hungarian Algorithm
      h_row, h_col = linear_sum_assignment(-iou_matrix)
      h_matrix = np.array(list(zip(h_row, h_col)))

      # Create new unmatched lists for old and new boxes
      matches = []

      # Go through the Hungarian Matrix,
      #  if matched element has IOU < threshold (0.4),
      #  add it to the matched list
      for h in h_matrix:
          if(iou_matrix[h[0], h[1]] > 0.4):
              matches.append(h.reshape(1, 2))

      if(len(matches) == 0):
          matches = np.empty((0, 2), dtype=int)
      else:
          matches = np.concatenate(matches, axis=0)
      return matches

  def get_bb_img(self, img, box):
      ''' return image arrray with bounding box dimensions'''

      shp = img.shape
      x1 = int(box[0] * shp[1] - box[2] * shp[1] * 0.5)  # center_x - width /2
      y1 = int(box[1] * shp[0] - box[3] * shp[0] * 0.5)  # center_y - height /2
      x2 = int(box[0] * shp[1] + box[2] * shp[1] * 0.5)  # center_x + width/2
      y2 = int(box[1] * shp[0] + box[3] * shp[0] * 0.5)  # center_y + height/2
      crop_img = img[y1:y2, x1:x2]
      return crop_img

  def check_bb_similarity(self, matched_bboxes, img1, pred_bb_1,
                          img2, pred_bb_2, ssim_bb_thres):
    ''' return the ssim score of associated bounding boxes'''

    for idx in range(0, matched_bboxes.shape[0]):

      bb_im_1 = get_bb_img(img1, pred_bb_1[matched_bboxes[idx, 0]].tolist())
      bb_im_2 = get_bb_img(img2, pred_bb_2[matched_bboxes[idx, 1]].tolist())

      # for ssim both the bbboxes should of same dimension
      # resizing second bbox to first one
      bb_im_2 = cv2.resize(bb_im_2, (bb_im_1.shape[1], bb_im_1.shape[0]),
                           interpolation=cv2.INTER_AREA)

      # get ssim_score
      (bb_score, _) = compare_ssim(bb_im_1, bb_im_2,
                                   multichannel=True, full=True)

      # the ssim score shoud exceed the threshold score to be similar
      if bb_score <= ssim_bb:
        # print('The BBs are not similar')
        return False

    return True
