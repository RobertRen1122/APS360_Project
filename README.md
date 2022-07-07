# APS360 Project
## Object Detection Process
### Data Processing
[ ]: Check the original dimension of each frame. Can I reshape them to 448 by 448?   
[ ]: If I reshape the pictures, the bounding box coordinates will need to be reshaped accordingly.  
[ ]: We need to reformat the csv file to have appropriate labels. `[is_person (optional), box_x, box_y, box_w, box_h]`  
[ ]: Need to have each frame as a training image and label. For each image, we need to have labels.  
[ ]: Helper functions implementation.

### Baseline 
**Performance:**  
Currently, we overfit on one camera angle `cam_0`.  
[ ]: YOLOv1 loss  
[ ]: YOLOv1 model  
[ ]: YOLOv1 training

### Primary Model (if time allows)
**Performance:**  
[ ]: YOLOv3 loss  
[ ]: YOLOv3 model  
[ ]: YOLOv3 training

## Person Re-identification Process:
### Data Processing
[ ]: Reformat all images to the same size

### Baseline
**Performance:**  
[ ]: Simple script to compare the difference between 2 images
[ ]: Consider 2 images same if above a pre-set threshold.

### Primary Model
**Performance:**  
[ ]: In progress

