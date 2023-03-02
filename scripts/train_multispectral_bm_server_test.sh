nohup python -u train_multispectral.py \
    --device cpu \
    --data data/multispectral/kaist.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.add-eoir-boxmix.yaml \
    --batch-size 4 \
    --epochs 1 \
    --workers 16 \
    --entity cvlabnam \
    --project /131_data/namgi/add_eoir/checkpoints/boxmix \
    --name multispectral_boxmix_kaist_test \
    > /131_data/namgi/add_eoir/logs/multispectral_boxmix_kaist_test.log 2>&1 &
disown -a