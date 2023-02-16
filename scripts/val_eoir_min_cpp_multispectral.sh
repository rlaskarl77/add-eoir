EPOCH=200
EXPERIMENTS=('v2.0_Multispectral_copypaste@0.5_sizeVar@8_posVar_100epoch')
for exp in "${EXPERIMENTS[@]}"
do
    for task in test val
    do
        for phase in eo ir
        do
            set="${exp:5:2}"
            python val.py \
                --device 0,1,2,3 \
                --data data/add-eoir-${phase}-test-segment.yaml \
                --weights copy_paste/${exp}/weights/best.pt \
                --batch-size 64 \
                --person-only \
                --task ${task} \
                --exist-ok \
                --save-json \
                --save-txt \
                --save-conf \
                --eval-tod \
                --project copy_paste \
                --name ${exp} \
                > logs/min/val_${task}_${phase^^}_${exp}.log 2>&1
        done
    done
done
