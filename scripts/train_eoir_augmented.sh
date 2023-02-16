EPOCH=200
for set in eo ir
do
    python -um torch.distributed.run --nproc_per_node 4 train.py \
        --device 0,1,2,3 \
        --data data/add-eoir-${set}.yaml \
        --weights yolov5l.pt \
        --hyp data/hyps/hyp.add-eoir-augmented.yaml \
        --batch-size 64 \
        --cache ram \
        --person-only \
        --exist-ok \
        --epochs $EPOCH \
        --entity cvlab_detection \
        --project ADD_augmentation \
        --name v2.0_${set^^}_yolov5l_bs64_augmented_cpp@0.5_${EPOCH}epoch \
        > logs/train_v2.0_${set^^}_yolov5l_bs64_augmented_cpp@0.5_${EPOCH}epoch.log 2>&1
done