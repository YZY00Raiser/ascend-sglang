 sglang_source_path=$(pwd)
          echo "Source code path: ${sglang_source_path}"
          ln -sf ${sglang_source_path} /root/sglang
          pip install tabulate
          # Install sgl-eval
          # shellcheck source=scripts/ci/utils/sgl_eval_ref.sh
          sglang_source_path=$(pwd)
          echo "Source code path: ${sglang_source_path}"
          . "${sglang_source_path}/scripts/ci/utils/sgl_eval_ref.sh"
          pip install "$SGL_EVAL_SPEC"
          source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
          source /usr/local/Ascend/nnal/atb/set_env.sh || true
          source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash || true
          source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash || true
