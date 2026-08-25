# A5编译命令
bash build.sh -a deepep Ascend930

# 编译完安装whl
pip install output/deep*

# 安装后导入验证
python -c "import deepep; print(deepep.__version__)"

cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so && cd -
