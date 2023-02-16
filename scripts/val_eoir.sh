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
    python val.py \
        --device 0,1,2,3 \
        --data data/add-eoir-${set,,}-test.yaml \
        --weights ADD_augmentation/${exp}/weights/best.pt \
        --batch-size 64 \
        --person-only \
        --exist-ok \
        --save-json \
        --save-txt \
        --save-conf \
        --project ADD_augmentation_val2 \
        --name ${exp} \
        > logs/val_${exp}.log 2>&1
done
