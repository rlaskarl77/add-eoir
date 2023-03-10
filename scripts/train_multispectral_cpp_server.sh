EPOCH=150
python -um torch.distributed.run --nproc_per_node 4 train_multispectral_add_eoir.py \
    --device 0,1,2,3 \
    --data data/multispectral/add-eoir-test-segment-server.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.add-eoir-cpp.yaml \
    --batch-size 128 \
    --person-only \
    --exist-ok \
    --epochs 100 \
    --entity cvlab_detection \
    --project copy_paste \
    --name v2.1_Multispectral_bs64_midsize_noocclusion_originaldist_copypaste@1.0_${EPOCH}epoch \
    > /131_data/namgi/add_eoir/logs/train_copy_paste/v2.1_Multispectral_bs64_midsize_noocclusion_originaldist_copypaste@1.0_${EPOCH}epoch 2>&1