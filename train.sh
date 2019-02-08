dataset_path=/media/Car/Carfusion_Dataset/train/

python3 carfusioncombined2coco.py --label_dir gt/ --image_dir images/ --output_dir $dataset_path --path_dir $dataset_path

CUDA_VISIBLE_DEVICES=0 python Detector/tools/train_net_step.py   --dataset keypoints_carfusion_fifth --cfg Detector/configs/e2e_keypoint_car_kgnn_R-50-FPN_1x.yaml  --load_ckpt /media/Car/Carfusion_Dataset/cache/Outputs/model_car.pth  --bs 1 --iter_size 2 --use_tfboard #--no_cuda

