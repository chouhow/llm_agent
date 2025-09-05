from openai import OpenAI
import sys


def test_with_openai_library():
    # 配置客户端指向你的 Flask 服务器
    client = OpenAI(
        api_key="your_openai_api_key_here",  # 与 Flask 应用中设置的一致
        base_url="http://localhost:5000/v1"  # 指向你的 Flask 服务器
    )

    try:
        # 创建聊天补全请求，启用流式传输
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "请用中文解释机器学习的基本概念"}
            ],
            stream=True
        )

        print("开始接收流式响应:\n")
        full_response = ""

        # 处理流式响应
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content

        print(f"\n\n完整响应: {full_response}")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    test_with_openai_library()