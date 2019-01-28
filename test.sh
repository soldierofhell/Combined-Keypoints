path=$3
path_save=$path 
out_file=kps_txt_maskrcnn
#/media/Car/data_detector/data/top_view/
#sudo mkdir $path_save/keypoints_txt_new/
CUDA_VISIBLE_DEVICES=$2 python Detector/tools/infer_simple.py --dataset keypoints_carfusion --cfg_person Detector/configs/e2e_keypoint_rcnn_R-50-FPN_1x.yaml --cfg_car Detector/configs/e2e_keypoint_car_rcnn_R-50-FPN_1x.yaml --load_ckpt_car model_car.pth --load_ckpt_person model_person.pth --image_dir $path/$1  --output_dir $path_save/$1/$out_file/ --visualize False
#CUDA_VISIBLE_DEVICES=$2 python Detector/tools/infer_simple.py --dataset keypoints_carfusion --cfg_person Detector/configs/e2e_keypoint_rcnn_R-50-FPN_1x.yaml --cfg_car Detector/configs/e2e_keypoint_car_rcnn_R-50-FPN_1x.yaml --load_ckpt_car /media/Car/Carfusion_Dataset/cache/Outputs/e2e_keypoint_car_rcnn_R-50-FPN_1x/Sep22-07-49-21_gpuserver3_step/ckpt/model_step89999.pth --load_ckpt_person model_person.pth --image_dir $path/$1  --output_dir $path_save/$1/$out_file/ --visualize False
#CUDA_VISIBLE_DEVICES=$2 python Detector/tools/infer_simple.py --dataset keypoints_carfusion --cfg_person Detector/configs/e2e_keypoint_rcnn_R-50-FPN_1x.yaml --cfg_car Detector/configs/e2e_keypoint_car_rcnn_R-50-FPN_1x.yaml --load_ckpt_car /media/Car/Carfusion_Dataset/cache/Outputs/e2e_keypoint_car_rcnn_R-50-FPN_1x/Sep18-20-25-02_gpuserver4_step/ckpt/model_step49999.pth --load_ckpt_person model_person.pth --image_dir $path  --output_dir $path_save/$1/keypoints_txt_new/ --visualize True
mkdir $path_save/$1/kps_txt_maskrcnntracked/
python tracking.py $path_save $1 $out_file 
#results/top_view/ 

