nohup python -u train_multispectral.py \
    --device cpu \
    --data data/multispectral/add-eoir-test-segment2.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.add-eoir-boxmix.yaml \
    --batch-size 4 \
    --exist-ok \
    --epochs 1 \
    --entity cvlabnam \
    --project /131_data/namgi/add_eoir/checkpoints/boxmix \
    --name multispectral_boxmix_test \
    > /131_data/namgi/add_eoir/logs/multispectral_boxmix_test.log 2>&1 &
disown -a