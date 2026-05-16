| Model / 模型 | Accuracy Threshold / 准确性阈值 | Latency Threshold (s) / 延迟阈值 (秒) | TP Size / TP 大小 |
|--------------|-------------------------------|-------------------------------------|-------------------|
| deepseek-ai/deepseek-vl2-small | 0.320 | 56.1 | 1 |
| deepseek-ai/Janus-Pro-7B | 0.285 | 40.3 | 1 |
| Efficient-Large-Model/NVILA-8B-hf | 0.270 | 56.7 | 1 |
| Efficient-Large-Model/NVILA-Lite-2B-hf | 0.270 | 23.8 | 1 |
| google/gemma-4-E4B-it | 0.26 | 15.0 | 1 |
| google/gemma-4-26B-A4B-it | 0.27 | 22.3 | 2 |
| google/gemma-4-31B-it | 0.28 | 25.5 | 2 |
| mistral-community/pixtral-12b | 0.360 | 16.6 | 1 |
| moonshotai/Kimi-VL-A3B-Instruct | 0.330 | 23.5 | 1 |
| openbmb/MiniCPM-o-2_6 | 0.330 | 29.5 | 1 |
| openbmb/MiniCPM-v-2_6 | 0.259 | 36.3 | 1 |
| OpenGVLab/InternVL2_5-2B | 0.300 | 18.0 | 1 |
| Qwen/Qwen2-VL-7B-Instruct | 0.310 | 83.3 | 1 |
| Qwen/Qwen2.5-VL-7B-Instruct | 0.330 | 31.9 | 1 |
| Qwen/Qwen3-VL-30B-A3B-Instruct | 0.29 | 37.0 | 2 |
| unsloth/Mistral-Small-3.1-24B-Instruct-2503 | 0.30 | 16.7 | 1 |
| XiaomiMiMo/MiMo-VL-7B-RL | 0.28 | 40.0 | 1 |
| zai-org/GLM-4.1V-9B-Thinking | 0.280 | 30.4 | 1 |
| zai-org/GLM-4.5V-FP8 | 0.26 | 34.0 | 2 |
test_rope_rocm

--enable-mfu-metrics

--default-priority-value

--dllm-algorithm

验证兼容 OpenAI 的嵌入 API 端点



./test/registered/layers/mamba/test_causal_conv1d.py
./test/registered/layers/mamba/test_mamba2_mixer.py
./test/registered/layers/mamba/test_mamba_ssm.py
./test/registered/layers/mamba/test_mamba_ssm_ssd.py
./test/registered/layers/test_fla_layernorm_guard.py
./test/registered/perf/test_bench_one_batch_1gpu.py
./test/registered/perf/test_bench_one_batch_2gpu.py
./test/registered/perf/test_bench_serving_1gpu_large.py
./test/registered/perf/test_bench_serving_1gpu_part1.py
./test/registered/perf/test_bench_serving_1gpu_part2.py
./test/registered/perf/test_bench_serving_2gpu.py
./test/registered/perf/test_dpsk_r1_fp4_4gpu_perf.py
./test/registered/perf/test_gpt_oss_4gpu_perf.py
./test/registered/perf/test_text_models_perf.py
./test/registered/perf/test_vlm_perf_5090.py
./test/registered/perf/test_vlms_perf.py




./test/registered/unit/layers/test_mamba_state_scatter_triton.py
./test/registered/unit/managers/test_prefill_adder.py
./test/registered/unit/managers/test_profile_merger_http_api.py
./test/registered/unit/mem_cache/test_mamba_unittest.py
./test/registered/unit/mem_cache/test_nsa_pool_host_unit.py
./test/registered/unit/mem_cache/test_radix_cache_slru_accuracy.py
./test/registered/unit/mem_cache/test_radix_cache_unit.py
./test/registered/unit/mem_cache/test_swa_unittest.py
./test/registered/unit/model_executor/test_model_hooks.py
./test/registered/unit/model_loader/test_modelopt_export.py
./test/registered/unit/model_loader/test_modelopt_loader.py
./test/registered/unit/utils/test_profile_merger.py
