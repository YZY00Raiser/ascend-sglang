export USE_ACCDATA=2
python -m sglang.launch_server \
--model-path /home/weights/Qwen/Qwen3-VL-235B-A22B-Instruct \
--attention-backend ascend \
--tp 16 \
--port 30000 \
--host 127.0.0.1 \
--trust-remote-code \
--mem-fraction-static 0.8 \
--disable-radix-cache \
--disable-cuda-graph

python -m sglang.test.run_eval \
    --eval-name mmmu \
    --num-examples 100 \
    --num-threads 64 \
    --max-tokens 30 \
    --port 30000 \
    --host 127.0.0.1 
