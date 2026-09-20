evalscope eval \
    --model /home/weights/DeepSeek-V4-Pro-0813-w4a8 \
    --api-url http://192.168.25.216:6677/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets gpqa_diamond \
    --eval-batch-size 128 \
    --generation-config '{
        "do_sample": true,
        "max_tokens": 125000,
        "seed": 3407,
        "top_p": 1,
        "top_k": 20,
        "temperature": 1,
        "n": 1,
        "repetition_penalty": 1.0,
        "timeout": 6000,
        "stream": true,
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": true
            }
        }
    }' \
    --ignore-errors \
    --work-dir ./gpqa/
