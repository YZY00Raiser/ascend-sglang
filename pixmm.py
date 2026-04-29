#!/usr/bin/env python3
"""
SGLang 多模态推理脚本 - 单机 PD 分离架构
基于 SGLang 仓库代码实现

PD 分离: Prefill (预填充) 和 Decode (解码) 分别运行在不同 GPU 上
"""

import subprocess
import time
import requests
import sys
import os


# ============ 配置 ============
MODEL_PATH = "/home/weights/Qwen/Qwen3-VL-8B-Instruct"

# Encoder 配置 (视觉编码)
ENCODER_PORT = 30000
ENCODER_GPU_ID = 4

# PD 分离配置
# Prefill 节点: 负责计算 attention, 生成 KV Cache
PREFILL_PORT = 30001
PREFILL_GPU_ID = 5

# Decode 节点: 负责 token 生成 (对外提供 API)
DECODE_PORT = 30002
DECODE_GPU_ID = 6

# Load Balancer 配置 (PD 分离需要 LB)
LB_PORT = 30003

# 传输后端配置
TRANSFER_BACKEND = "mooncake"  # 可选: mooncake, nixl, ascend, fake, mori
IB_DEVICE = None  # 如果有 RDMA 设备，设置为 "mlx5_0" 等


def start_encoder_server():
    """启动 Encoder 服务器 (视觉编码)"""
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--encoder-only",
        "--encoder-transfer-backend", "zmq_to_scheduler",
        "--port", str(ENCODER_PORT),
        "--enable-prefix-mm-cache",
        "--base-gpu-id", str(ENCODER_GPU_ID),
    ]
    print(f"[Encoder] 启动服务器 (端口 {ENCODER_PORT}, GPU {ENCODER_GPU_ID})...")
    print(f"Command: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def start_prefill_server():
    """启动 Prefill 服务器 (PD 分离中的 P 节点)"""
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--language-only",
        "--encoder-urls", f"http://127.0.0.1:{ENCODER_PORT}",
        "--encoder-transfer-backend", "zmq_to_scheduler",
        "--port", str(PREFILL_PORT),
        "--base-gpu-id", str(PREFILL_GPU_ID),
        # PD 分离关键参数 (参考仓库代码)
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", TRANSFER_BACKEND,
        "--tp", "1",
    ]
    
    if IB_DEVICE:
        cmd.extend(["--disaggregation-ib-device", IB_DEVICE])
    
    print(f"[Prefill] 启动服务器 (端口 {PREFILL_PORT}, GPU {PREFILL_GPU_ID})...")
    print(f"Command: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def start_decode_server():
    """启动 Decode 服务器 (PD 分离中的 D 节点)"""
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--language-only",
        "--encoder-urls", f"http://127.0.0.1:{ENCODER_PORT}",
        "--encoder-transfer-backend", "zmq_to_scheduler",
        "--port", str(DECODE_PORT),
        "--base-gpu-id", str(DECODE_GPU_ID),
        # PD 分离关键参数 (参考仓库代码)
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", TRANSFER_BACKEND,
        "--tp", "1",
    ]
    
    if IB_DEVICE:
        cmd.extend(["--disaggregation-ib-device", IB_DEVICE])
    
    print(f"[Decode] 启动服务器 (端口 {DECODE_PORT}, GPU {DECODE_GPU_ID})...")
    print(f"Command: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def start_load_balancer():
    """启动 PD 分离专用的 Load Balancer"""
    cmd = [
        "python", "-m", "sglang_router.launch_router",
        "--pd-disaggregation",
        "--mini-lb",
        "--prefill", f"http://127.0.0.1:{PREFILL_PORT}",
        "--decode", f"http://127.0.0.1:{DECODE_PORT}",
        "--host", "127.0.0.1",
        "--port", str(LB_PORT),
    ]
    print(f"[LoadBalancer] 启动负载均衡器 (端口 {LB_PORT})...")
    print(f"Command: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def wait_for_server(url, timeout=300):
    """等待服务器启动"""
    print(f"等待服务器 {url} 启动...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"[✓] 服务器 {url} 已就绪!")
                return True
        except:
            pass
        time.sleep(2)
    raise TimeoutError(f"服务器 {url} 启动超时")


def send_request(
    text: str = "这张图里有什么？",
    image_url: str = "https://miaobi-lite.bj.bcebos.com/miaobi/5mao/b%27b2Ny6K%2BG5Yir5Luj56CBXzE3MzQ2MzcyNjAuMzgxNDk5NQ%3D%3D%27/0.png",
    temperature: float = 0.5,
    max_new_tokens: int = 600
):
    """发送推理请求到 Load Balancer"""
    headers = {"Content-Type": "application/json"}
    
    data = {
        "text": text,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        },
        "image_data": image_url
    }

    try:
        res = requests.post(
            f"http://127.0.0.1:{LB_PORT}/generate",
            json=data,
            headers=headers,
            timeout=120
        )
        res.raise_for_status()
        response_data = res.json()
        return response_data
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None


def main():
    """主函数：启动 PD 分离架构的服务器并发送请求"""
    encoder_process = None
    prefill_process = None
    decode_process = None
    lb_process = None
    
    try:
        # 1. 启动 Encoder 服务器
        encoder_process = start_encoder_server()
        wait_for_server(f"http://127.0.0.1:{ENCODER_PORT}")
        
        # 2. 启动 Prefill 服务器
        prefill_process = start_prefill_server()
        wait_for_server(f"http://127.0.0.1:{PREFILL_PORT}")
        
        # 3. 启动 Decode 服务器
        decode_process = start_decode_server()
        wait_for_server(f"http://127.0.0.1:{DECODE_PORT}")
        
        # 4. 启动 Load Balancer (PD 分离必需)
        lb_process = start_load_balancer()
        wait_for_server(f"http://127.0.0.1:{LB_PORT}")
        
        print("\n" + "="*60)
        print("PD 分离架构启动完成!")
        print(f"  Encoder:  GPU {ENCODER_GPU_ID},  端口 {ENCODER_PORT}")
        print(f"  Prefill:  GPU {PREFILL_GPU_ID},  端口 {PREFILL_PORT}")
        print(f"  Decode:   GPU {DECODE_GPU_ID},  端口 {DECODE_PORT}")
        print(f"  LB:       端口 {LB_PORT} (API 端点)")
        print("="*60 + "\n")
        
        # 5. 发送请求
        print("发送推理请求...")
        result = send_request(
            text="这张图里有什么？",
            image_url="https://miaobi-lite.bj.bcebos.com/miaobi/5mao/b%27b2Ny6K%2BG5Yir5Luj56CBXzE3MzQ2MzcyNjAuMzgxNDk5NQ%3D%3D%27/0.png"
        )
        
        if result:
            print("\n=== 推理结果 ===")
            print(result)
        
        # 保持运行
        print("\n服务器正在运行，按 Ctrl+C 停止...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n用户中断，正在停止服务器...")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理进程
        if lb_process:
            lb_process.terminate()
            print("[LoadBalancer] 已停止")
        if decode_process:
            decode_process.terminate()
            print("[Decode] 服务器已停止")
        if prefill_process:
            prefill_process.terminate()
            print("[Prefill] 服务器已停止")
        if encoder_process:
            encoder_process.terminate()
            print("[Encoder] 服务器已停止")


def quick_request_only():
    """仅发送请求（假设服务器已在运行）"""
    print("发送推理请求...")
    result = send_request(
        text="这张图里有什么？",
        image_url="https://miaobi-lite.bj.bcebos.com/miaobi/5mao/b%27b2Ny6K%2BG5Yir5Luj56CBXzE3MzQ2MzcyNjAuMzgxNDk5NQ%3D%3D%27/0.png"
    )
    
    if result:
        print("\n=== 推理结果 ===")
        print(result)


def start_servers_only():
    """仅启动服务器，不发送请求"""
    encoder_process = None
    prefill_process = None
    decode_process = None
    lb_process = None
    
    try:
        # 1. 启动 Encoder 服务器
        encoder_process = start_encoder_server()
        wait_for_server(f"http://127.0.0.1:{ENCODER_PORT}")
        
        # 2. 启动 Prefill 服务器
        prefill_process = start_prefill_server()
        wait_for_server(f"http://127.0.0.1:{PREFILL_PORT}")
        
        # 3. 启动 Decode 服务器
        decode_process = start_decode_server()
        wait_for_server(f"http://127.0.0.1:{DECODE_PORT}")
        
        # 4. 启动 Load Balancer
        lb_process = start_load_balancer()
        wait_for_server(f"http://127.0.0.1:{LB_PORT}")
        
        print("\n" + "="*60)
        print("PD 分离架构启动完成!")
        print(f"  Encoder:  GPU {ENCODER_GPU_ID},  端口 {ENCODER_PORT}")
        print(f"  Prefill:  GPU {PREFILL_GPU_ID},  端口 {PREFILL_PORT}")
        print(f"  Decode:   GPU {DECODE_GPU_ID},  端口 {DECODE_PORT}")
        print(f"  LB:       端口 {LB_PORT} (API 端点)")
        print("="*60 + "\n")
        
        # 保持运行
        print("服务器正在运行，按 Ctrl+C 停止...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n用户中断，正在停止服务器...")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理进程
        if lb_process:
            lb_process.terminate()
            print("[LoadBalancer] 已停止")
        if decode_process:
            decode_process.terminate()
            print("[Decode] 服务器已停止")
        if prefill_process:
            prefill_process.terminate()
            print("[Prefill] 服务器已停止")
        if encoder_process:
            encoder_process.terminate()
            print("[Encoder] 服务器已停止")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SGLang 多模态推理脚本 - 单机 PD 分离")
    parser.add_argument(
        "--mode",
        choices=["full", "request-only", "servers-only"],
        default="full",
        help="""
        运行模式:
          full          - 启动所有服务器 + 发送请求
          request-only  - 仅发送请求 (服务器已运行)
          servers-only  - 仅启动服务器 (不发送请求)
        """
    )
    parser.add_argument(
        "--transfer-backend",
        choices=["mooncake", "nixl", "ascend", "fake", "mori"],
        default="mooncake",
        help="PD 分离传输后端 (默认: mooncake)"
    )
    parser.add_argument(
        "--ib-device",
        type=str,
        default=None,
        help="RDMA 设备 (如 mlx5_0)，不设置则使用默认"
    )
    
    args = parser.parse_args()
    
    # 更新配置
    TRANSFER_BACKEND = args.transfer_backend
    IB_DEVICE = args.ib_device
    
    if args.mode == "full":
        main()
    elif args.mode == "request-only":
        quick_request_only()
    else:
        start_servers_only()
