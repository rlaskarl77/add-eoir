EPOCH=1
python -u train_add_eoir.py \
    --device cpu \
    --data data/add-eoir-ir2.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.add-eoir-augmented.yaml \
    --batch-size 64 \
    --person-only \
    --exist-ok \
    --workers 32 \
    --epochs $EPOCH \
    --entity cvlab_detection \
    --project /131_data/namgi/add_eoir/checkpoints/mixup \
    --name v2.1_IR_prepare_cache \
    > /131_data/namgi/add_eoir/logs/train_v2.1_IR_prepare_cache.log 2>&1