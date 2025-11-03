# tscratch – From-scratch BPE + Transformer (multi-file)

## Layout
```
tscratch/
  __init__.py
  bpe_tokenizer.py
  data.py
  model.py
  utils.py
  train.py
main.py
```

## Quick start
Train tokenizer + model in one go:
```
python main.py --train_txt ./shakespeare.txt   --vocab_size 1024 --seq_len 256   --d_model 512 --heads 8 --layers 6 --ff_mult 4   --batch_size 32 --steps 10000 --lr 3e-4 --warmup 200   --sample_every 500 --save ckpt.pt
```

Resume training:
```
python main.py --resume ckpt.pt --steps 20000 --save ckpt.pt
```

Generate a sample at the end is automatic.
