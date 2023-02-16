nohup python -um torch.distributed.run --nproc_per_node 8 train_multispectral.py \
    --device 0,1,2,3,4,5,6,7 \
    --data data/multispectral/kaist.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.kaist-multispectral.yaml \
    --batch-size 96 \
    --exist-ok \
    --cache ram \
    --epochs 200 \
    --entity cvlabnam \
    --project /131_data/namgi/add_eoir/checkpoints/kaist \
    --name multispectral_copypaste@0.5_eclipse_200epochs \
    > /131_data/namgi/add_eoir/logs/multispectral_copypaste@0.5_eclipse_200epochs.log 2>&1 &
disown -a