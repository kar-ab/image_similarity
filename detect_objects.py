from yolov4.tf import YOLOv4
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2

class DetectObjects(object):
  def __init__(self, classes_path, weights_path):
    '''Yolov4 trained on COCO dataset'''
    
    self.yolo = YOLOv4()
    self.yolo.classes = classes_path
    self.yolo.make_model()
    self.yolo.load_weights(weights_path, weights_type="yolo")	


  def yolo_coco(self, img): 
    '''Run Inference on images using YOLOv4 pre-trained on COCO'''

    resized_image = self.yolo.resize_image(img)
    resized_image = resized_image / 255.
    input_data = resized_image[np.newaxis, ...].astype(np.float32)
    candidates = self.yolo.model.predict(input_data)
    
    _candidates = []
    result = img.copy()

    for candidate in candidates:
        grid_size = candidate.shape[1]
        _candidates.append(tf.reshape(candidate,
                                      shape=(1, grid_size * grid_size * 3, -1)))
        #candidates.shape == Dim(batch, candidates, (bbox))
        candidates = np.concatenate(_candidates, axis=1)
        #pred_bboxes.shape == (candidates, (x, y, w, h, class_id, prob))
        pred_bboxes = self.yolo.candidates_to_pred_bboxes(candidates[0], 
                                                     iou_threshold=0.40,
                                                     score_threshold=0.60)
        # removing rows with only zereos
        pred_bboxes = pred_bboxes[~(pred_bboxes==0).all(1)] 
        pred_bboxes = self.yolo.fit_pred_bboxes_to_original(pred_bboxes, img.shape)
        bb_image = self.yolo.draw_bboxes(img, pred_bboxes)
        bb_image = cv2.cvtColor(bb_image, cv2.COLOR_BGR2RGB)
    return bb_image, pred_bboxes

