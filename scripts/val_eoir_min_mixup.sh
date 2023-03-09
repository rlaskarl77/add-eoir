EXPERIMENTS=('v2.1_EO_bs64_midsize_mixup@1.0__100epoch'
'v2.1_IR_bs64_midsize_mixup@1.0__100epoch')
for exp in "${EXPERIMENTS[@]}"
do
    phase="${exp:5:2}"

    if [ phase="EO" ]
    then
        imgsz=1024
        echo "phase=EO, ${imgsz}"
    elif [ phase="IR" ]
    then
        imgsz=640
        echo "phase=IR, ${imgsz}"
    else
        echo "phase is not valid: ${phase}\n with experiment ${exp}"
        exit 0
    fi

    for task in test val
    do
        python val_add_eoir.py \
            --device 0,1,2,3 \
            --data data/add-eoir-${phase,,}-segment.yaml \
            --weights mixup/${exp}/weights/best.pt \
            --batch-size 64 \
            --imgsz ${imgsz} \
            --task ${task} \
            --exist-ok \
            --save-json \
            --save-txt \
            --save-conf \
            --eval-tod \
            --project mixup_val \
            --name ${exp}_${task}_all \
            > logs/val_mixup/${task}_${phase}_all_${exp}.log 2>&1
    done
done
