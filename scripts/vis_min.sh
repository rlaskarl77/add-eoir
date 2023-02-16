nohup python -u vis_min.py \
    --data 'add-eoir-eo-test-segment.yaml' \
    --weights 'copy_paste/EO_yolov5l_copypaste@0.5_200epoch/weights/best.pt' \
    --batch-size 64 \
    --conf-thres 0.1 \
    --iou-thres 0.6 \
    --device 'cpu' \
    --workers 8 \
    --project vis_min_test_eo \
    --exist-ok \
    --line-thickness 2 \
    --with-conf \
    > logs/vis_min/EO_base_test_yolov5l_wo_train.log 2>&1 &
disown -a