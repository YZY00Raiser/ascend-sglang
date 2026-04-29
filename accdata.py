#!/usr/bin/env python3
"""
将 accdata.sh 转换为 Python 脚本
用于启动 sglang 服务并运行评估
"""

import os
import subprocess
import sys
import time
import socket
import threading
import re


def check_port_open(host, port, timeout=1):
    """检查端口是否开放"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def wait_for_service(host, port, max_wait=300, check_interval=2):
    """
    等待服务启动成功
    :param host: 主机地址
    :param port: 端口号
    :param max_wait: 最大等待时间（秒）
    :param check_interval: 检查间隔（秒）
    :return: 是否成功启动
    """
    print(f"等待服务在 {host}:{port} 上就绪...")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if check_port_open(host, port):
            elapsed = time.time() - start_time
            print(f"服务已成功启动！耗时: {elapsed:.1f} 秒")
            return True
        time.sleep(check_interval)
        elapsed = time.time() - start_time
        print(f"  等待中... {elapsed:.1f}s / {max_wait}s")

    print(f"服务启动超时（超过 {max_wait} 秒）")
    return False


def run_command_with_capture(cmd, env=None):
    """运行命令并捕获输出"""
    print(f"执行命令: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)

    process.wait()
    return process.returncode, ''.join(output_lines)


def extract_score_from_output(output):
    """从输出中提取 Score 值"""
    # 匹配 "Score: 0.360" 或类似的格式
    match = re.search(r'Score:\s*(0\.\d+)', output)
    if match:
        return float(match.group(1))
    return None


def assert_mmmu_accuracy(score, expected=0.36, tolerance=0.01):
    """
    断言 mmmu 精度值与期望值在允许范围内
    :param score: 实际精度值
    :param expected: 期望精度值（默认 0.36）
    :param tolerance: 允许误差范围（默认 1%，即 0.01）
    """
    if score is None:
        raise AssertionError("未能从输出中提取到 mmmu 精度值")

    lower_bound = expected * (1 - tolerance)
    upper_bound = expected * (1 + tolerance)

    print(f"\n{'=' * 60}")
    print(f"mmmu 精度值断言检查:")
    print(f"  期望值: {expected}")
    print(f"  实际值: {score}")
    print(f"  允许范围: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"  误差范围: ±{tolerance * 100}%")
    print(f"{'=' * 60}")

    if not (lower_bound <= score <= upper_bound):
        raise AssertionError(
            f"mmmu 精度值 {score} 超出允许范围 [{lower_bound:.4f}, {upper_bound:.4f}], "
            f"期望值 {expected} ± {tolerance * 100}%"
        )

    print("✓ 精度值检查通过！")
    print("OK")
    return True


def stream_output(process, prefix=""):
    """实时输出子进程的输出"""
    for line in process.stdout:
        print(f"{prefix}{line}", end='')


def main():
    host = '127.0.0.1'
    port = 30000

    # 设置环境变量
    env = os.environ.copy()
    env['USE_ACCDATA'] = '2'

    # 启动 sglang 服务
    launch_cmd = [
        'python', '-m', 'sglang.launch_server',
        '--model-path', '/home/weights/Qwen/Qwen3-VL-235B-A22B-Instruct',
        '--attention-backend', 'ascend',
        '--tp', '16',
        '--port', str(port),
        '--host', host,
        '--trust-remote-code',
        '--mem-fraction-static', '0.8',
        '--disable-radix-cache',
        '--disable-cuda-graph'
    ]

    print("=" * 60)
    print("启动 sglang 服务...")
    print("=" * 60)
    print(f"命令: {' '.join(launch_cmd)}")
    print(f"环境变量: USE_ACCDATA={env.get('USE_ACCDATA')}")
    print(f"工作目录: {os.getcwd()}")
    print("-" * 60)

    # 使用 Popen 启动服务（后台运行）
    server_process = subprocess.Popen(
        launch_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    # 启动线程实时输出服务日志
    log_thread = threading.Thread(
        target=stream_output,
        args=(server_process, "[Server] "),
        daemon=True
    )
    log_thread.start()

    # 等待服务启动成功
    print("-" * 60)
    service_ready = wait_for_service(host, port, max_wait=600, check_interval=2)
    print("-" * 60)

    if not service_ready:
        print("服务启动失败，正在终止...")
        server_process.terminate()
        server_process.wait()
        sys.exit(1)

    # 运行评估
    eval_cmd = [
        'python', '-m', 'sglang.test.run_eval',
        '--eval-name', 'mmmu',
        '--num-examples', '100',
        '--num-threads', '64',
        '--max-tokens', '30',
        '--port', str(port),
        '--host', host
    ]

    print("\n" + "=" * 60)
    print("运行评估...")
    print("=" * 60)

    eval_returncode, output = run_command_with_capture(eval_cmd)

    # 提取 mmmu 精度值并进行断言检查
    if eval_returncode == 0:
        score = extract_score_from_output(output)
        try:
            assert_mmmu_accuracy(score, expected=0.36, tolerance=0.01)
        except AssertionError as e:
            print(f"\n✗ 断言失败: {e}")
            eval_returncode = 1

    # 终止服务
    print("\n" + "=" * 60)
    print("终止服务...")
    print("=" * 60)
    server_process.terminate()
    server_process.wait()

    sys.exit(eval_returncode)


if __name__ == '__main__':
    main()
