#!/usr/bin/env python3
"""
将 accdata.sh 转换为 Python 脚本
用于启动 sglang 服务并运行评估
"""

import os
import subprocess
import sys
import time


def run_command(cmd, env=None):
    """运行命令并实时输出"""
    print(f"执行命令: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    for line in process.stdout:
        print(line, end='')

    process.wait()
    return process.returncode


def main():
    # 设置环境变量
    env = os.environ.copy()
    env['USE_ACCDATA'] = '2'

    # 启动 sglang 服务
    launch_cmd = [
        'python', '-m', 'sglang.launch_server',
        '--model-path', '/home/weights/Qwen/Qwen3-VL-235B-A22B-Instruct',
        '--attention-backend', 'ascend',
        '--tp', '16',
        '--port', '30000',
        '--host', '127.0.0.1',
        '--trust-remote-code',
        '--mem-fraction-static', '0.8',
        '--disable-radix-cache',
        '--disable-cuda-graph'
    ]

    print("=" * 60)
    print("启动 sglang 服务...")
    print("=" * 60)

    # 使用 Popen 启动服务（后台运行）
    server_process = subprocess.Popen(
        launch_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    # 等待服务启动
    print("等待服务启动...")
    time.sleep(30)

    # 运行评估
    eval_cmd = [
        'python', '-m', 'sglang.test.run_eval',
        '--eval-name', 'mmmu',
        '--num-examples', '100',
        '--num-threads', '64',
        '--max-tokens', '30',
        '--port', '30000',
        '--host', '127.0.0.1'
    ]

    print("\n" + "=" * 60)
    print("运行评估...")
    print("=" * 60)

    eval_returncode = run_command(eval_cmd)

    # 终止服务
    print("\n" + "=" * 60)
    print("终止服务...")
    print("=" * 60)
    server_process.terminate()
    server_process.wait()

    sys.exit(eval_returncode)


if __name__ == '__main__':
    main()
