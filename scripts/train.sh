nohup python -um torch.distributed.run --nproc_per_node 4 train.py \
    --device 0,1,2,3 \
    --data data/add-eoir-ir.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.add-eoir-augmented.yaml \
    --batch-size 64 \
    --cache ram \
    --person-only \
    --exist-ok \
    --epochs 200 \
    --entity cvlab_detection \
    --project ADD_augmentation \
    --name v2.0_IR_yolov5l_bs64_augmented_mixup@0.3_cpp@0.3_200epoch \
    > logs/train_v2.0_IR_yolov5l_bs64_augmented_mixup@0.3_cpp@0.3_200epoch.log 2>&1 &
disown -a