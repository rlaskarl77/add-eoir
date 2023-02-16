nohup python -um torch.distributed.run --nproc_per_node 4 train_multispectral.py \
    --device 0,1,2,3 \
    --data /home/namgi/yolov5_latest/data/multispectral/add-eoir-test-segment.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.add-eoir-cpp.yaml \
    --batch-size 64 \
    --person-only \
    --exist-ok \
    --epochs 100 \
    --entity cvlabnam \
    --project copy_paste \
    --name v2.0_Multispectral_copypaste@0.5_sizedist_posVar_100epoch \
    > logs/train_copy_paste/v2.0_train_Multispectral_copypaste@0.5_sizedist_posVar_100epoch.log 2>&1 &
disown -a