PROJECT='/131_data/namgi/add_eoir/checkpoints/copy_paste'
EXPERIMENTS=('v2.1_Multispectral_bs256_midsize_noocclusion_originaldist_copypaste@1.0_150epoch')
for exp in "${EXPERIMENTS[@]}"
do
    for task in test val
    do
        echo "task=${task}"
        for phase in "eo" "ir"
        do
            if [ "$phase" = "eo" ]
            then
                imgsz=1024
                echo "phase=EO, ${imgsz}"
            else
                imgsz=640
                echo "phase=IR, ${imgsz}"
            fi
            for ponly in "false" "true"
            do
                if [ "$ponly" = "true" ]
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
                    --data data/add-eoir-${phase}-segment.yaml \
                    --weights ${PROJECT}/${exp}/weights/best.pt \
                    --batch-size 64 \
                    ${ponly_scr} \
                    --task ${task} \
                    --imgsz ${imgsz} \
                    --exist-ok \
                    --save-json \
                    --save-txt \
                    --save-conf \
                    --eval-tod \
                    --project copy_paste_val \
                    --name ${exp}_${phase^^}_${task}_${ponly_str} \
                    > logs/copy_paste_val/${exp}_${phase^^}_${task}_${ponly_str}.log 2>&1
            done
        done
    done
done
