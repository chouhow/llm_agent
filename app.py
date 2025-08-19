import json
import os

from openai import OpenAI

import db_tools
from logger import log_model_response

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_all_tables",
            "description": "获取数据库中所有表的名称列表。不需要任何参数，直接返回所有表名。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_query_data",
            "description": "执行SQL查询语句并返回结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "需要执行的SQL查询语句"
                    }
                },
                "required": ["sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": "获取指定表的创建结构信息（CREATE TABLE语句）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "需要获取结构的数据库表名称"
                    }
                },
                "required": ["table_name"]
            }
        }
    }
]

def run_agent():
    messages = [
        {
            "role": "system",
            "content": """你是一位数据库专家，擅长使用SQL语句进行数据库查询和操作。
            根据用户的需求，生成并执行相应的SQL语句。
            """
        }
    ]

    # 持续获取用户输入直到Ctrl+C退出
    while True:
        try:
            user_input = input("请输入内容（按Ctrl+C退出）: ")
            messages.append({"role": "user", "content": user_input})
            while True:
                response = client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3",  # 可以根据需要更换为其他模型
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                response_message = response.choices[0].message
                messages.append(response_message)
                log_model_response(response_message, model="deepseek-ai/DeepSeek-V3")
                print(response_message)
                if response_message.content:
                    print(response_message.content)

                if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                    # 处理工具调用
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        # 根据工具名称调用对应的db_tools函数
                        if function_name == "get_all_tables":
                            result = db_tools.get_all_tables()
                        elif function_name == "get_query_data":
                            result = db_tools.get_query_data(**function_args)
                        elif function_name == "get_table_schema":
                            result = db_tools.get_table_schema(**function_args)
                        else:
                            result = {"error": f"未知函数: {function_name}"}

                        print(result)

                        # 将工具调用结果添加到消息历史
                        tool_message = {
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(result, cls=DecimalEncoder),
                            "tool_call_id": tool_call.id
                        }
                        messages.append(tool_message)
                    continue
                else:
                    break
        except KeyboardInterrupt:
            print("\n程序已退出。")
            break

if __name__ == '__main__':

    run_agent()