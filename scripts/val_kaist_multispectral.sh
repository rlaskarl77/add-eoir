EXPERIMENTS=('multispectral_copypaste@0.5_ellipse_200epochs3'
'multispectral_copypaste@0.5_ellipse_200epochs4')
PROJECT="/131_data/namgi/add_eoir/checkpoints/kaist"
for exp in "${EXPERIMENTS[@]}"
do
    for task in val
    do
        for phase in rgb ir
        do
            set="${exp:5:2}"
            python val.py \
                --device 0,1,2,3 \
                --data data/kaist-rgbt-${phase}.yaml \
                --weights /131_data/namgi/add_eoir/checkpoints/kaist/${exp}/weights/best.pt \
                --batch-size 64 \
                --task ${task} \
                --save-json \
                --save-txt \
                --save-conf \
                --project /131_data/namgi/add_eoir/checkpoints/kaist_val \
                --name ${exp} \
                > /131_data/namgi/add_eoir/logs/val/${task}_${phase}_${exp}.log 2>&1
        done
    done
done
