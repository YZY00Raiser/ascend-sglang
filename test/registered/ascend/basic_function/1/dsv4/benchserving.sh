unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

export WP=/home/weights/DeepSeek-V4-Flash-0731-w8a8

# host/port 指向 router 地址（与 py 用例中 self.host/self.port 对应）
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --dataset-name random \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 2400 \
  --max-concurrency 800 \
  --random-range-ratio 1 \
  --request-rate inf \
  --seed 1 \
  --model $WP \
  --warmup-requests 16
