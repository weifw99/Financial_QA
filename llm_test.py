import litellm

# 加载配置文件
litellm.load_config("/path/to/config.yaml")

# 打印配置信息，检查 API 密钥是否正确
print(f"API Key: {litellm.api_key}")

try:
    # 调用嵌入模型
    response = litellm.embedding(
        model="text-embedding-bge-m3",
        input="这是一段测试文本。"
    )
    print(response['data'][0]['embedding'])
except Exception as e:
    print(f"An error occurred: {e}")