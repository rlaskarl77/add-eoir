EPOCH=200
EXPERIMENTS=('v2.0_IR_yolov5l_bs64_mixup@0.5_200epoch'
'v2.0_IR_yolov5l_bs64_cpp@0.5_200epoch'
'v2.0_IR_yolov5l_bs64_100epoch'
'v2.0_IR_testval_bs64_mixup@0.5_cpp@0.5_200epoch'
'v2.0_IR_testval_bs64_mixup@0.5_200epoch'
'v2.0_IR_testval_bs64_cpp@0.5_200epoch'
'v2.0_EO_yolov5l_bs64_mixup@0.5_200epoch'
'v2.0_EO_yolov5l_bs64_cpp@0.5_200epoch'
'v2.0_EO_yolov5l_bs64_100epoch'
'v2.0_EO_testval_bs64_mixup@0.5_cpp@0.5_200epoch'
'v2.0_EO_testval_bs64_mixup@0.5_200epoch'
'v2.0_EO_testval_bs64_cpp@0.5_200epoch')
for exp in "${EXPERIMENTS[@]}"
do
    set="${exp:5:2}"
    cp -f /home/namgi/yolov5_latest/ADD_augmentation_val/${exp}/best_predictions.json /home/namgi/yoloeval/results/yolo${set}/${exp}.json
done
