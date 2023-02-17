nohup python -um torch.distributed.run --nproc_per_node 4 train_multispectral.py \
    --device 0,1,2,3 \
    --data data/multispectral/kaist.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.kaist-multispectral.yaml \
    --batch-size 48 \
    --exist-ok \
    --cache ram \
    --epochs 200 \
    --entity cvlabnam \
    --project /131_data/namgi/add_eoir/checkpoints/kaist \
    --name local_multispectral_copypaste@0.5_ellipse_200epochs \
    > /131_data/namgi/add_eoir/logs/local_multispectral_copypaste@0.5_ellipse_200epochs.log 2>&1 &
disown -a