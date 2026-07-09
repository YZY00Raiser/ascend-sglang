#!/bin/bash
set -euo pipefail

# 日志路径配置
LOG_DIR=/data/y/glm5_2_double_w8a8_logs
# 按日期分割日志，避免单文件膨胀
LOG_FILE="${LOG_DIR}/bench_serving_$(date +%Y%m%d).log"

# 强制创建日志目录
#mkdir -p "${LOG_DIR}"

# 循环压测
while true; do
    start_time=$(date +%s)
    echo "=============================================" >> "${LOG_FILE}"
    echo "【压测启动】$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
    echo "日志文件: ${LOG_FILE}" >> "${LOG_FILE}"
    echo "目标服务: 172.22.3.71:6688" >> "${LOG_FILE}"
    echo "=============================================" >> "${LOG_FILE}"

    # 执行压测，捕获退出码
    python -m sglang.bench_serving \
        --dataset-name random \
        --backend sglang \
        --model /root/.cache/modelscope/hub/models/Eco-Tech/GLM-5.2-w8a8 \
        --dataset-path /data/yzy/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
        --host 172.22.3.71 \
        --port 6688 \
        --max-concurrency 128 \
        --random-input-len 3500 \
        --random-output-len 1500 \
        --num-prompts 128 \
        --random-range-ratio 1 \
    ret_code=$?

    end_time=$(date +%s)
    cost=$(( end_time - start_time ))

    # 记录结束信息、返回码、耗时
    echo "---------------------------------------------" >> "${LOG_FILE}"
    echo "【本轮结束】$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
    echo "执行耗时: ${cost}s | 进程退出码: ${ret_code}" >> "${LOG_FILE}"
done
