import os
import glob
import cv2
import numpy as np
import pickle
from skimage.metrics import structural_similarity as compare_ssim

# from otherfile import TheClass
# theclass = TheClass()
# theclass.run()  

from detect_objects import DetectObjects
from bb_utils import bb_utils

class ImageSimilarity(object):

  def __init__(self, dataset_path, classes_path, weights_path):
    self.image_list = sorted(glob.glob(dataset_path))
    self.list_obj_det = []
    self.bb_cnt_unq = []
    self.grp_list = []
    self.classes_path = classes_path
    self.weights_path = weights_path

  def group_list(self):
    ''' to group list according to the number of BB '''
    self.bb_cnt_unq = set(map(lambda x:x[1], self.list_obj_det))
    self.grp_list = [[y[0] for y in self.list_obj_det if y[1] == x]
                     for x in self.bb_cnt_unq]

  def run_obj_det(self):
    ''' run object detection '''
    
    for idx, path in enumerate(self.image_list):
      im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
      obj_det = Detectobjects(self.classes_path, self.weights_path)
      _, pred_bb = obj_det.yolo_coco(im)
      list_obj_det.append([idx, len(pred_bb), pred_bb])
    return list_obj_det


  def compare_two_images(self, idx_1, idx_2, ssim_im_thres, ssim_bb_thres):
    img1 = cv2.cvtColor(cv2.imread(self.image_list[idx_1]), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(self.image_list[idx_2]), cv2.COLOR_BGR2RGB)

    det_obj = DetectObjects(self.classes_path, self.weights_path)

    _, pred_bb_1 = det_obj.yolo_coco(img1)
    _, pred_bb_2 = det_obj.yolo_coco(img2)
    ssim_score, result_bb_ssim = self.get_similarity(img1, img2,
                                            pred_bb_1, pred_bb_1, 
                                            ssim_bb_thres)

    if (result_bb_ssim):
      if (ssim_score > ssim_im_thres):
        print('Images are similar')
    else:
      print('BBs are not similar, so Images are also not similar')
      ssim_score = 0
    return ssim_score

  def get_similarity(self, img1, img2, pred_bboxes_1, pred_bboxes_2, ssim_bb_thres):
    ''' get similarity score for given images'''

    matched_bboxes = bb_utils().associate_bounding_boxes(pred_bboxes_1, pred_bboxes_2)

    (ssim_imgs, diff_imgs) = compare_ssim(img1, img2,
                                          multichannel=True,
                                          full=True)
    result_bb_ssim = False

    if (matched_bboxes.shape[0] == max(len(pred_bboxes_1),
                                       len(pred_bboxes_2))):
      # print("all BB are associated, checking ssim of all associated BB")
      result_bb_ssim = bb_utils().check_bb_similarity(matched_bboxes,
                                           img1, pred_bboxes_1,
                                           img2, pred_bboxes_2, ssim_bb_thres)

    return ssim_imgs, result_bb_ssim

  def Query_search(self, qry_idx, ssim_im_thres, ssim_bb_thres):
    ''' search similar images in dataset for given query_index '''

    self.group_list()
    qry_bb_cnt = self.list_obj_det[qry_idx][1]
    qry_bb = self.list_obj_det[qry_idx][2]
    mat_bb_idx = self.grp_list[list(self.bb_cnt_unq).index(qry_bb_cnt)]


    img1 = cv2.cvtColor(cv2.imread(self.image_list[qry_idx]),
                        cv2.COLOR_BGR2RGB)

    similar_idx_list = []

    for idx in mat_bb_idx:
      img2 = cv2.cvtColor(cv2.imread(self.image_list[idx]),
                          cv2.COLOR_BGR2RGB)
      next_grp_bb = self.list_obj_det[idx][2]
      ssim_score, result_bb_ssim = self.get_similarity(img1, img2,
                                              qry_bb, next_grp_bb, 
                                              ssim_bb_thres)

      if (result_bb_ssim):
        if (ssim_score > ssim_im_thres):
          # print('Images are similar', idx)
          similar_idx_list.append(idx)
      else:
        # print('BBs are not similar, so Images are also not similar', idx)
        ssim_score = 0

    if len(similar_idx_list) == 1:
      print('The queried image index is unique')
    return similar_idx_list

  def deduplicate_dataset(self, ssim_im_thres, ssim_bb_thres):

    self.group_list()
    unique_grp_list = self.grp_list
    print('Before de-duplication')
    print(unique_grp_list)
    print('\n\n')

    # iterating all set of bb_groups
    for each_grp_idx, each_grp in enumerate(unique_grp_list):

      print('Group contents before de-duplication')
      print(each_grp)
      print('length of group', len(each_grp))
      print('\n')

      list_similar_imgs = []
      for bb_grp_idx, query_index in enumerate(each_grp):

        qry_bb_cnt = self.list_obj_det[query_index][1]
        qry_bb = self.list_obj_det[query_index][2]

        print('Querying for Index:', query_index)
        img1 = cv2.cvtColor(cv2.imread(self.image_list[query_index]),
                            cv2.COLOR_BGR2RGB)
        similar_idx_list = []

        # iterate over all the next indexes in the same bb group
        for _, next_grp_idx in enumerate(each_grp[bb_grp_idx+1:]):

          img2 = cv2.cvtColor(cv2.imread(self.image_list[next_grp_idx]),
                              cv2.COLOR_BGR2RGB)
          next_grp_bb = self.list_obj_det[next_grp_idx][2]
          ssim_score, result_bb_ssim = self.get_similarity(img1, img2,
                                                  qry_bb, next_grp_bb,
                                                  ssim_bb_thres)

          if (result_bb_ssim):
            # print('Similar Image indixes', next_grp_idx)
            similar_idx_list.append(next_grp_idx)
            each_grp.remove(next_grp_idx)

        if len(similar_idx_list) == 0:
            print('Image index ', query_index,' is a unique image')
        else:
            print('Following Images were found similar to a Image no. ',
                  query_index, 'and were removed from group', similar_idx_list)

      unique_grp_list[each_grp_idx] = each_grp
      print('Group contents after de-duplication')
      print(each_grp)
      print('length of group', len(each_grp))
      print('\n')

    print('\n\n')
    print('After de-duplication')
    print('Unique Images(indices):', unique_grp_list)

