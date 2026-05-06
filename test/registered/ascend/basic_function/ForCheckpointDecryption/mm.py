import requests

# 请求头
headers = {"Content-Type": "application/json"}

# 请求数据（正确格式化）
data = {
    "text": "这张图里有什么？",
    "sampling-params": {
        "temperature": 0.5,
        "max_new_tokens": 600
    },
    "image_data": "https://miaobi-lite.bj.bcebos.com/miaobi/5mao/b%27b2Ny6K%2BG5Yir5Luj56CBXzE3MzQ2MzcyNjAuMzgxNDk5NQ%3D%3D%27/0.png"
}

try:
    # 发送 POST 请求到本地服务
    res = requests.post("http://127.0.0.1:30002/generate", json=data, headers=headers, verify=False)
    res.raise_for_status()  # 检查请求是否成功
    response_data = res.json()
    print(response_data)
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
