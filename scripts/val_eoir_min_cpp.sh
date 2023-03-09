PROJECT='copy_paste'
EXPERIMENTS=('v2.1_EO_bs64_midsize_noocclusion_originaldist_copypaste@1.0_100epoch')
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
        for ponly in "false" "true"
        do
            if [ "$ponly" == "true" ]
            then
                ponly_str='ponly'
                ponly_scr='--person-only'
                echo $ponly_str
            else
                ponly_str='all'
                ponly_scr=''
                echo $ponly_str
            fi
            python val_add_eoir.py \
                --device 0,1,2,3 \
                --data data/add-eoir-${phase,,}-segment.yaml \
                --weights ${PROJECT}/${exp}/weights/best.pt \
                --batch-size 64 \
                --imgsz ${imgsz} \
                --task ${task} \
                --exist-ok \
                --save-json \
                --save-txt \
                ${ponly_scr} \
                --save-conf \
                --eval-tod \
                --project copy_paste_val \
                --name ${exp}_${task}_${ponly_str} \
                > logs/copy_paste_val/${task}_${phase}_${ponly_str}_${exp}.log 2>&1
        done
    done
done