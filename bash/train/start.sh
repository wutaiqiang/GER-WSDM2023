cd Ger

#sleep 2h

python3 -u ger/train.py \
    --dataset_path data/zeshel \
    --pretrained_model bert-base-uncased \
    --name hgat_dualloss_mu01_lr2 \
    --log_dir /Ger/logs1 \
    --mu 0.1 \
    --epoch 10 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --encode_batch_size 128 \
    --eval_interval 200 \
    --logging_interval 10 \
    --graph \
    --gnn_layers 3 \
    --learning_rate 2e-5 \
    --do_eval \
    --do_test \
    --do_train \
    --data_parallel \
    --dual_loss \
    --handle_batch_size 4 \
    --return_type hgat 

#--dual_loss \ --train_ratio 0.6 \
