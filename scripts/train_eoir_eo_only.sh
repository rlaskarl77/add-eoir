EPOCH=100
python -um torch.distributed.run --nproc_per_node 4 train_add_eoir.py \
    --device 0,1,2,3 \
    --data data/add-eoir-eo-segment.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.add-eoir-augmented.yaml \
    --batch-size 64 \
    --person-only \
    --exist-ok \
    --workers 24 \
    --epochs $EPOCH \
    --entity cvlab_detection \
    --project mixup \
    --name v2.1_EO_bs64_midsize_mixup@1.0__${EPOCH}epoch \
    > logs/train_mixup/train_v2.1_EO_bs64_midsize_mixup@1.0_${EPOCH}epoch.log 2>&1