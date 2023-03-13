EPOCH=150
BATCH_SIZE=128
python -um torch.distributed.run --nproc_per_node 4 train_add_eoir.py \
    --device 0,1,2,3 \
    --data data/add-eoir-eo-segment-server.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.add-eoir-cpp-mixup.yaml \
    --batch-size ${BATCH_SIZE} \
    --workers 48 \
    --person-only \
    --exist-ok \
    --cache ram \
    --epochs $EPOCH \
    --entity cvlab_detection \
    --project /131_data/namgi/add_eoir/checkpoints/copy_paste \
    --name v2.1_EO_bs${BATCH_SIZE}_midsize_noocclusion_originaldist_copypaste_mixup@1.0_${EPOCH}epoch \
    > /131_data/namgi/add_eoir/logs/copy_paste_mixup/train_v2.1_EO_bs${BATCH_SIZE}_midsize_noocclusion_originaldist_copypaste_mixup@1.0_${EPOCH}epoch.log 2>&1