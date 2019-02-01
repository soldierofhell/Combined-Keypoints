dataset_path=/home/minh2/ShapeDeformModeling/BestDetector/Detector/data/carfusion/
dataset_path=/media/Car/Carfusion_Dataset/train/

#python3 carfusioncombined2coco.py --label_dir gt/ --image_dir images/ --output_dir $dataset_path --path_dir $dataset_path

#rm /media/Car/Carfusion_Dataset/cache/keypoints_carfusion_coco_gt_roidb.pkl

#python3 visualize.py 
CUDA_VISIBLE_DEVICES=0 python Detector/tools/train_net_step.py   --dataset keypoints_carfusion_fifth --cfg Detector/configs/e2e_keypoint_car_kgnn_R-50-FPN_1x.yaml  --load_ckpt /media/Car/Carfusion_Dataset/cache/Outputs/model_car.pth  --bs 1 --iter_size 2 --use_tfboard #--no_cuda









#CUDA_VISIBLE_DEVICES=0,1,2,3 python Detector/tools/train_net_step.py   --dataset keypoints_carfusion_coco --cfg Detector/configs/e2e_keypoint_car_rcnn_R-50-FPN_1x.yaml --load_ckpt /media/Car/Carfusion_Dataset/cache/Outputs/model_car.pth   --bs 8 --iter_size 2 --use_tfboard
#python Detector/tools/train_net_step.py   --dataset keypoints_carfusion_coco --cfg Detector/configs/e2e_faster_rcnn_R-50-FPN_1x.yaml --bs 8 --iter_size 2 --use_tfboard
#python Detector/tools/test_net.py --multi-gpu-testing --dataset keypoints_carfusion_penn --cfg Detector/configs/e2e_keypoint_car_rcnn_R-50-FPN_1x.yaml --load_ckpt /home/minh2/ShapeDeformModeling/carfusion/Keypoint_detections/Outputs/fifth_trained/model_car.pth
#python tools/train_net_step.py   --dataset keypoints_carfusion --cfg mask-rcnn.pytorch/configs/e2e_keypoint_rcnn_R-50-FPN_1x.yaml   --bs 8 --iter_size 2 --use_tfboard
#python tools/train_net_step.py   --dataset keypoints_coco2017 --cfg configs/e2e_keypoint_rcnn_R-50-FPN_1x.yaml   --bs 8 --iter_size 2 --use_tfboard


