# image_similarity
Remove Similar images from dataset

pip3 install -r requirements.txt

Download files required for yolov4

! gdown --id 15P4cYyZ2Sd876HKAEWSmeRdFl_j-0upi 
! wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names


Note:

for ssim_im_thres = 0.8 and ssim_bb_thres = 0.5, 96 unique iamges were obtained

for more fine tuned results, set higher values of thresholds. 

Acknowledgement: 
For yolo and bounding box association, code has been borrowed from : 
https://github.com/Jeremy26/visual_fusion_course/blob/main/Visual_Fusion_Late.ipynb
