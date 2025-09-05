from flask import Flask, request, Response, jsonify
import json
import time
from openai import OpenAI  # 使用 openai >= 1.0.0

app = Flask(__name__)

# 假设的 API 密钥，实际应从安全的环境变量或配置管理中读取
API_KEY = "your_openai_api_key_here"


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    # 简单的授权验证（生产环境应更严谨）
    auth_header = request.headers.get('Authorization')
    if not auth_header or auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Invalid authorization"}), 401

    # 解析请求 JSON
    data = request.get_json()
    model = data.get('model', 'gpt-3.5-turbo')
    messages = data.get('messages', [])
    stream = data.get('stream', False)

    # 如果不是流式请求，可以返回错误或实现非流式处理
    if not stream:
        return jsonify({"error": "This endpoint currently only supports stream=True"}), 400

    def generate_stream():
        # 示例 1：模拟流式输出
        simulated_response = "这是一个模拟的流式响应，逐词返回结果。"
        words = list(simulated_response)

        # 首先发送一个包含角色信息的 chunk
        base_chunk = {
            "id": "chatcmpl-simulated123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(base_chunk, ensure_ascii=False)}\n\n"

        # 逐词生成内容
        for word in words:
            chunk = {
                "id": "chatcmpl-simulated123",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            time.sleep(0.1)  # 模拟延迟

        # 发送结束信号
        end_chunk = {
            "id": "chatcmpl-simulated123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

        # 示例 2：若要真实转发 OpenAI 的流（取消注释并替换 client 配置）
        # client = OpenAI(api_key=API_KEY)  # 或你的 base_url
        # try:
        #     stream = client.chat.completions.create(
        #         model=model,
        #         messages=messages,
        #         stream=True
        #     )
        #     for chunk in stream:
        #         # 直接转发从 OpenAI 收到的 chunk
        #         # 注意：可能需要根据 OpenAI 最新 SDK 返回结构微调
        #         yield f"data: {chunk.json()}\n\n"
        #     yield "data: [DONE]\n\n"
        # except Exception as e:
        #     # 错误处理：尝试以 SSE 格式发送错误信息
        #     error_chunk = {
        #         "error": {"message": f"An error occurred: {str(e)}"},
        #         "choices": [{"delta": {}}]  # 保持结构相似
        #     }
        #     yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')


if __name__ == '__main__':
    # 生产环境应使用 debug=False, 并考虑使用更好的 WSGI 服务器
    app.run(debug=True, port=5000)