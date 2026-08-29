python3 tests/python/deepep/test_fused_deep_moe_a5.py \
  --quant fp8_e4m3 \
  --num-processes 4 \
  --hidden 7168 \
  --moe-intermediate-size 3072 \
  --num-topk 8 \
  --num-experts 32 \
  --num-warmups 3 \
  --num-tests 10 \
  --weight-format NZ
