
python3 -u ger/cross_encoder.py \
    --dataset_path data/zeshel \
    --pretrained_model bert-base-uncased \
    --name bert_only@64_end_1gpu \
    --log_dir logs2 \
    --epoch 3 \
    --train_batch_size 1 \
    --top_k 64 \
    --eval_batch_size 36 \
    --eval_interval 400 \
    --logging_interval 100 \
    --learning_rate 1e-5 \
    --data_parallel \
    --train_path  @path\
    --valid_path  @path \
    --test_path @path
