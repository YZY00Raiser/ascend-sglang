# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
# export SGLANG_NPU_USE_MULTI_STREAM=1
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=enp23s0f3
export GLOO_SOCKET_IFNAME=enp23s0f3
export TRANSFORMERS_VERBOSITY=error

MODEL_PATH=/root/.cache/modelscope/hub/models/Eco-Tech/GLM-5.2-w8a8
export SGLANG_NPU_PROFILING=0
export SGLANG_NPU_PROFILING_BS=16
#export PYTHONPATH=/sgl-workspace/sglang/python:$PYTHONPATH

LOG_DIR=/data/y/glm5_2_double_w8a8_logs
mkdir -p $LOG_DIR

SCRIPT_LOG="$LOG_DIR/glm5_2_double_w8a8_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$SCRIPT_LOG") 2>&1

echo "=== Script started at $(date) ==="
echo "=== Master log: $SCRIPT_LOG ==="

D_IP=('172.22.3.71' '172.22.3.77')
LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
#export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
#export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
      LOG_FILE="$LOG_DIR/launch_server_${D_IP[$i]}_$(date +%Y%m%d_%H%M%S).log"
      echo "=== Starting launch_server on ${D_IP[$i]} at $(date) ==="
      echo "=== Server log: $LOG_FILE ==="
      python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --host ${D_IP[$i]} \
        --dist-init-addr 172.22.3.71:50000  \
        --tp-size 32 \
        --nnodes 2 --node-rank $i \
        --dp-size 32 \
        --enable-dp-attention \
        --chunked-prefill-size -1 \
        --max-prefill-tokens 28672 \
        --trust-remote-code \
        --mem-fraction-static 0.8 \
        --served-model-name GLM-5.2-w8a8 \
        --cuda-graph-bs 1 \
        --max-running-requests 32 \
        --quantization modelslim \
        --speculative-draft-model-quantization unquant \
        --moe-a2a-backend deepep --deepep-mode auto \
        --load-balance-method round_robin \
        --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2  \
        --port 6688 --tokenizer-worker-num 32 \
        > "$LOG_FILE" 2>&1
        NODE_RANK=$i
        break
    fi
done


