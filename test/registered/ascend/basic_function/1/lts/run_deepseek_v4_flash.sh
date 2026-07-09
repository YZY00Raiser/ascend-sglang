source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

current_date=$(date +%Y%m%d%H%M%S)


nohup python3 sglang/test/manual/ascend/lts/test_npu_lts_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py > log/test_ascend_lts_qwen3_coder_next_${current_date}.log 2>&1 &
