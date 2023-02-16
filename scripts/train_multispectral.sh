nohup python -um torch.distributed.run --nproc_per_node 4 train_multispectral.py \
    --device 0,1,2,3 \
    --data /home/namgi/yolov5_latest/data/multispectral/add-eoir-test.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.add-eoir-msm.yaml \
    --batch-size 64 \
    --person-only \
    --exist-ok \
    --epochs 1 \
    --entity cvlabnam \
    --project Multispectral_augmentation \
    --name v2.0_Multispectral_yolov5l_bs64_mcm@0.5_test \
    > logs/train_v2.0_Multispectral_yolov5l_bs64_mcm@0.5_test.log 2>&1 &
disown -a