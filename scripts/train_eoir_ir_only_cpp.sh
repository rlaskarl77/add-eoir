EPOCH=100
python -um torch.distributed.run --nproc_per_node 4 train_add_eoir.py \
    --device 0,1,2,3 \
    --data data/add-eoir-ir-segment.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.add-eoir-cpp.yaml \
    --batch-size 64 \
    --person-only \
    --exist-ok \
    --workers 24 \
    --epochs $EPOCH \
    --entity cvlab_detection \
    --project copy_paste \
    --name v2.1_IR_bs64_midsize_noocclusion_originaldist_copypaste@1.0_${EPOCH}epoch \
    > logs/train_copy_paste/train_v2.1_IR_bs64_midsize_noocclusion_originaldist_copypaste@1.0_${EPOCH}epoch.log 2>&1