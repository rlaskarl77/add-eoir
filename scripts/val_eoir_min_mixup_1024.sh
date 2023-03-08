EXPERIMENTS=('v2.1_EO_bs64_midsize_mixup@1.0__100epoch'
'v2.1_IR_bs64_midsize_mixup@1.0__100epoch')
for exp in "${EXPERIMENTS[@]}"
do
    phase="${exp:5:2}"
    for task in test val
    do
        python val_add_eoir.py \
            --device 0,1,2,3 \
            --data data/add-eoir-${phase,,}-segment.yaml \
            --weights mixup/${exp}/weights/best.pt \
            --batch-size 64 \
            --imgsz 1024 \
            --person-only \
            --task ${task} \
            --exist-ok \
            --save-json \
            --save-txt \
            --save-conf \
            --eval-tod \
            --project mixup_val \
            --name ${exp} \
            > logs/val_mixup/val_${task}_${phase}_${exp}.log 2>&1
    done
done
