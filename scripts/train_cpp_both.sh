for mod in IR, EO
do
    python -um torch.distributed.run --nproc_per_node 4 train.py \
        --device 0,1,2,3 \
        --data data/add-eoir-${mod,,}-test-segment.yaml \
        --weights yolov5l.pt \
        --hyp data/hyps/hyp.add-eoir-cpp.yaml \
        --batch-size 64 \
        --person-only \
        --exist-ok \
        --epochs 200 \
        --entity cvlabnam \
        --project copy_paste \
        --name v2.0_${mod}_yolov5l_copypaste@0.5_sizeVar@8.0_posVar_200epoch \
        > logs/train_copy_paste/v2.0_train_${mod}_yolov5l_copypaste@0.5_sizeVar@8.0_posVar_200epoch.log 2>&1
done