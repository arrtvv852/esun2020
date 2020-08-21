bt=$(date +"%r")

echo "" >> logs.txt
echo "==================" >> log.txt
echo "Begin time : $bt" >> log.txt

# bt = $(date +%s)
# echo "$bt"

# train
python joint_bert_crf.py \
  --task_name=customized \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --do_export=true \
  --data_dir=./data-v1_0804/ \
  --train_batch_size=2 \
  --num_train_epochs=10 \
  --max_seq_length=512 \
  --learning_rate=3e-5 \
  --vocab_file=./bert_wwm/vocab.txt \
  --bert_config_file=./bert_wwm/bert_config.json \
  --init_checkpoint=./bert_wwm/bert_model.ckpt \
  --output_dir=./output-v1_0804/ \
  --export_dir=./output-v1_0804/export/ \
  --alpha=1.0

et=$(date +"%r")
echo "End time : $et" >> log.txt