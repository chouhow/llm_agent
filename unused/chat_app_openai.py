from pathlib import Path

import streamlit as st
import time
import json
import os
from openai import OpenAI
from decimal import Decimal

import db_tools
from logger import log_model_response
from dotenv import load_dotenv


# project_root = Path(__file__).resolve().parent.parent
# env_file_path = project_root / '.env.27'
# print(env_file_path)
# load_dotenv(dotenv_path=env_file_path)

load_dotenv()

print(os.getenv("OPENAI_API_KEY"))
print(os.getenv("OPENAI_API_BASE"))

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
CHAT_MODEL = os.getenv("CHAT_MODEL")


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

# 页面配置
st.set_page_config(
    page_title="LLM对话助手",
    page_icon="💬",
    layout="wide"
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": """你是一位数据库专家，擅长使用SQL语句进行数据库查询和操作。
            根据用户的需求，生成并执行相应的SQL语句。
            """
        }
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# 清除对话函数
def clear_chat():
    st.session_state.messages = []
    st.session_state.chat_history = []



# 页面标题和说明
st.title("💬 LLM对话助手")


# 显示清除对话按钮
st.sidebar.button("清除对话历史", on_click=clear_chat, type="primary")


# 显示对话历史
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        for component in message["components"]:
            if component["type"] == "code":
                st.code(component["content"])
            elif component["type"] == "text":
                st.text(component["content"])
            elif component["type"] == "title":
                st.title(component["content"])
            elif component["type"] == "error":
                st.error(component["content"])
            elif component["type"] == "subheader":
                st.subheader(component["content"])
            elif component["type"] == "dataframe":
                st.dataframe(component["content"])
            elif component["type"] == "table":
                st.dataframe(component["content"])
            elif component["type"] == "markdown":
                st.markdown(component["content"])
            elif component["type"] == "image":
                st.image(component["content"])
            else:
                st.write(component["content"])


# 处理用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 将用户输入添加到消息列表
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示用户输入
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "components": [{"type": "text", "content": prompt}]})

    while True:
        # 生成LLM回复
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            print(f"请求Base URL: {client.base_url}")
            response = client.chat.completions.create(
                model=CHAT_MODEL,  # 可以根据需要更换为其他模型
                messages=st.session_state.messages,
                tools=db_tools.tools,
                tool_choice="auto"
            )
            response_message = response.choices[0].message
            if response_message.content:
                message_placeholder.markdown(response_message.content)

            history_components=[]
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                # 处理工具调用
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    print(tool_call.function.arguments)
                    function_args = json.loads(tool_call.function.arguments)

                    # 根据工具名称调用对应的db_tools函数
                    if function_name == "get_all_tables":
                        st.markdown("### 调用工具：get_all_tables")
                        history_components.append({"type": "markdown", "content": "### 调用工具：get_all_tables"})
                        result = db_tools.get_all_tables()
                        if "error" in result:
                            st.error(f"错误: {result['error']}")
                            history_components.append({"type": "error", "content": result['error']})
                        else:
                            st.write("数据库表列表:")
                            history_components.append({"type": "text", "content": "数据库表列表:"})
                            st.code("\n".join(result["tables"]))
                            history_components.append({"type": "code", "content": result['tables']})
                    elif function_name == "get_query_data":
                        st.markdown("### 调用工具：get_query_data")
                        history_components.append({"type": "markdown", "content": "### 调用工具：get_query_data"})
                        # 显示执行的SQL语句
                        st.write("执行SQL:")
                        history_components.append({"type": "text", "content": "执行SQL:"})
                        st.code(function_args.get("sql", ""))
                        history_components.append({"type": "code", "content": function_args.get("sql", "")})
                        result = db_tools.get_query_data(**function_args)
                        if "error" in result:
                            st.error(f"查询错误: {result['error']}")
                            history_components.append({"type": "error", "content": result['error']})
                        else:
                            st.subheader("查询结果:")
                            history_components.append({"type": "text", "content": "查询结果:"})
                            st.dataframe(result["data"])
                            history_components.append({"type": "dataframe", "content": result["data"]})
                    elif function_name == "get_table_schema":
                        st.markdown("### 调用工具：get_table_schema")
                        history_components.append({"type": "markdown", "content": "### 调用工具：get_table_schema"})
                        table_name = function_args.get("table_name", "")
                        st.write(f"表结构: {table_name}")
                        history_components.append({"type": "text", "content": table_name})
                        result = db_tools.get_table_schema(**function_args)
                        if "error" in result:
                            st.error(f"获取结构错误: {result['error']}")
                            history_components.append({"type": "error", "content": result['error']})
                        else:
                            st.code(result["schema"])
                            history_components.append({"type": "code", "content": result["schema"]})
                    else:
                        result = {"error": f"未知函数: {function_name}"}
                        st.error(result["error"])
                        history_components.append({"type": "error", "content": result["error"]})

                    print(result)

                    # 将工具调用结果添加到消息历史
                    tool_message = {
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(result, cls=DecimalEncoder),
                        "tool_call_id": tool_call.id
                    }
                    st.session_state.messages.append(tool_message)
                    st.session_state.chat_history.append({"role": "assistant", "components": history_components})
                continue
            else:
                break


