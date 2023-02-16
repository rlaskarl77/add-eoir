python -um torch.distributed.run --nproc_per_node 4 train.py \
    --device 0,1,2,3 \
    --data data/add-eoir-ir-test-segment.yaml \
    --weights yolov5l.pt \
    --hyp data/hyps/hyp.add-eoir-cpp.yaml \
    --batch-size 64 \
    --person-only \
    --exist-ok \
    --epochs 1 \
    --entity cvlabnam \
    --project test_copy_paste \
    --name IR_test2 \
    > logs/test_copy_paste/IR_test2.log 2>&1 &