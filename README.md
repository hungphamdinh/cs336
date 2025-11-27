python -m trans.main \
  --train_txt tinystories/tinystories_train.txt \
  --vocab_size 2048 \
  --seq_len 128 \
  --d_model 256 \
  --heads 4 \
  --layers 4 \
  --batch_size 32 \
  --steps 4000 \
  --lr 3e-4 \
  --warmup 200 \
  --sample_every 500 \
  --save ./runs/small-128c256h4l4/ckpt.pt

python -m trans.main \
  --generate \
  --ckpt ./runs/small-128c256h4l4/ckpt.pt \
  --tokenizer tokenizer.json \
  --gen_prompt "Once upon a time in Hue," \
  --max_new_tokens 200 \
  --device cuda

python -m trans.main \
  --estimate-training \
  --vocab_size 2048 \
  --seq_len 128 \
  --d_model 256 \
  --heads 4 \
  --layers 4 \
  --batch_size 32 \
  --steps 4000 \


python -m trans.main \
  --estimate-training \
  --vocab_size 1024 --seq_len 256 \
  --d_model 384 --heads 6 --layers 6 --ff_mult 4 \
  --batch_size 32 --steps 2000 \
  --device auto

python -m trans.main \
  --train_txt tinystories/tinystories_train.txt \
  --benchmark \
  --warmup_steps 5 \
  --measure_steps 10

python -m trans.main \
  --train_txt tinystories/tinystories_train.txt \
  --profile \
  --profile_steps 20

python -m trans.main \
  --train_txt tinystories/tinystories_train.txt \
  --profile_precision \
  --profile_steps 20 \
  --precision_mode bf16

 python -m trans.main \
  --train_txt tinystories/tinystories_train.txt \
  --profile_precision \
  --profile_steps 20 \
  --precision_mode bf16
