LOG_DIR=/data/y/glm5_2_double_w8a8_logs
LOG_FILE="$LOG_DIR/bench_serving.log"

while true; do
    echo "=== Starting bench_serving at $(date) ===" >> "$LOG_FILE"
    echo "=== Bench log: $LOG_FILE ===" >> "$LOG_FILE"
    python -m sglang.bench_serving \
    --dataset-name random --backend sglang \
    --model /root/.cache/modelscope/hub/models/Eco-Tech/GLM-5.2-w8a8 \
    --dataset-path /data/y/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --host 172.22.3.71 --port 6688 --max-concurrency 128 \
    --random-input-len 3500 --random-output-len 1500 \
    --num-prompts 128 --random-range-ratio 1 \
    >> "$LOG_FILE" 2>&1
done
