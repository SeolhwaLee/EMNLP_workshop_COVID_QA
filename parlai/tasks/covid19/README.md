Task: covid
==============
Description: covid-19 QA data and a poly-encoder model

Tags: #covid

1.Download files (including train&valid data and model)
```
git clone https://github.com/qli74/ParlAI
cd ParlAI
python examples/display_data.py -t covid -n 1
```

2.web chat on localhost (use default port: 35496)
```
./start_browser_service.sh
```
or change 'localhost' to ip address in PariAI/parlai/chat_service/services/browser_chat/client.py _run_browser()
![example](https://github.com/qli74/ParlAI/blob/master/cov1.png)


3.terminal chat
```
./start_terminal_service.sh
```
![example](https://github.com/qli74/ParlAI/blob/master/cov2.png)


4.another api file written with fastapi: ParlAI/fastapi_covid.py\
https://github.com/qli74/ParlAI/blob/master/fastapi_covid.py

5.code for training & eval & interactive
```
python examples/train_model.py --init-model zoo:pretrained_transformers/model_poly/model -t covid \
  --model transformer/polyencoder --batchsize 5 --eval-batchsize 10 \
  --warmup_updates 100 --optimizer admax --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 \
  -lr 2e-03 --data-parallel True --history-size 20 --label-truncate 100 \
  --text-truncate 360 --num-epochs 20.0 --max_train_time 200000 -veps -1 \
  -vme 8000 --validation-metric accuracy --validation-metric-mode max \
  --save-after-valid False --log_every_n_secs 20 --candidates batch --fp16 True \
  --dict-tokenizer bpe --dict-lower True --output-scaling 0.06 \
  --variant xlm --reduction-type mean --share-encoders False \
  --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 \
  --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 \
  --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 \
  --learn-embeddings True --polyencoder-type codes --poly-n-codes 64 \
  --poly-attention-type basic --dict-endtoken __start__ \
  --dict-file  data/models/pretrained_transformers/model_poly/model.dict \
  --model-file model/poly/covid7 -ttim 10000 -stim 200
```
```
python examples/eval_model.py -m transformer/polyencoder -mf model/poly/covid7 -t covid --encode-candidate-vecs true --eval-candidates fixed  
```
```
python examples/interactive.py -m transformer/polyencoder -mf model/poly/covid7 --encode-candidate-vecs true --single-turn True
```
