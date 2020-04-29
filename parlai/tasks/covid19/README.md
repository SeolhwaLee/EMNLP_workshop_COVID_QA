Task: covid
==============
Description: covid-19 QA data and a poly-encoder model

Tags: #covid19

** Fine-tuned model download **
```
https://drive.google.com/file/d/1yDF7mYmVed-BoTkfjuRSLNw69UXeSU1S/view?usp=sharing
```
Download the model and decompress & put the Parlai_ver2/fine_tune_model/

1.Download files (including train&valid data and model)
```
git clone https://github.com/sseol11/Parlai_ver2.git
cd ParlAI_ver2
python examples/display_data.py -t covid19 -n 1
```

2.web chat on localhost (use default port: 35496)
```
./start_browser_service.sh
```
or change 'localhost' to ip address in Parlai_ver2/parlai/chat_service/services/browser_chat/client.py _run_browser()
![example](https://github.com/qli74/ParlAI/blob/master/cov1.png)


3.terminal chat
```
./start_terminal_service.sh
```
![example](https://github.com/qli74/ParlAI/blob/master/cov2.png)


4.another api file written with fastapi: Parlai_ver2/fastapi_covid.py\
https://github.com/sseol11/Parlai_ver2/blob/master/fastapi_covid.py

5.code for training & eval & interactive
```
python -u examples/train_model.py \
  --init-model zoo:pretrained_transformers/poly_model_huge_reddit/model \
  -t covid19 \
  --model transformer/polyencoder --batchsize 64 --eval-batchsize 10 \
  --warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 \
  -lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 \
  --text-truncate 360 --num-epochs 30.0 --max_train_time 200000 -veps 0.5 \
  -vme 8000 --validation-metric accuracy --validation-metric-mode max \
  --save-after-valid True --log_every_n_secs 20 --candidates batch --fp16 True \
  --dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 \
  --variant xlm --reduction-type mean --share-encoders False \
  --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 \
  --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 \
  --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 \
  --learn-embeddings True --polyencoder-type codes --poly-n-codes 64 \
  --poly-attention-type basic --dict-endtoken __start__ \
  --model-file fine_tuning_model/covid19_scraped_ver6/poly_encoder_covid19 \
  --dict-file zoo:pretrained_transformers/poly_model_huge_reddit/model.dict \
  --image-mode no_image_model
```
```
python examples/eval_model.py -m transformer/polyencoder -mf fine_tuning_model/covid19_scraped_ver6/poly_encoder_covid19 -t covid19 --encode-candidate-vecs true --eval-candidates fixed  
```
```
python examples/interactive.py -m transformer/polyencoder -mf fine_tuning_model/covid19_scraped_ver6/poly_encoder_covid19 --encode-candidate-vecs true --single-turn True
```
