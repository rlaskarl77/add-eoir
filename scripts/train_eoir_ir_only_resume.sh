EPOCH=100
python -um torch.distributed.run --nproc_per_node 4 train_add_eoir.py \
    --device 0,1,2,3 \
    --data data/add-eoir-ir2.yaml \
    --resume \
    --hyp data/hyps/hyp.add-eoir-augmented.yaml \
    --batch-size 64 \
    --person-only \
    --exist-ok \
    --workers 24 \
    --epochs $EPOCH \
    --entity cvlab_detection \
    --project /131_data/namgi/add_eoir/checkpoints/mixup \
    --name v2.1_IR_bs64_mixup@1.0_${EPOCH}epoch \
    > /131_data/namgi/add_eoir/logs/train_v2.1_IR_bs64_mixup@1.0_${EPOCH}epoch.log 2>&1