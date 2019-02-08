dataset_path=./Detector/data/fifth/

python3 carfusion2coco.py --label_dir gt/ --image_dir images_jpg/ --output_dir . --path_dir $dataset_path

CUDA_VISIBLE_DEVICES=0 python Detector/tools/train_net_step.py   --dataset keypoints_carfusion_fifth --cfg Detector/configs/e2e_keypoint_car_rcnn_R-50-FPN_1x.yaml  --load_ckpt model_car.pth  --bs 1 --iter_size 2 --use_tfboard #--no_cuda

