
from pathlib import Path

import streamlit as st

import json
import os

from decimal import Decimal
from datetime import date, datetime

from dotenv import load_dotenv
from langchain_core.tools import tool

from langchain_openai.chat_models import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store
from pydantic import SecretStr
from langchain_community.chat_models import ChatTongyi
from langgraph.prebuilt import create_react_agent
from mysql.connector import Error
import mysql.connector

from tools import print_langgraph_result


# project_root = Path(__file__).resolve().parent.parent
# env_file_path = project_root / '.env.27'
# print(env_file_path)
# load_dotenv(dotenv_path=env_file_path)

def create_db_connection():
    """创建MySQL数据库连接"""
    try:
        conn = mysql.connector.connect(
            host='192.168.100.27',
            user='zmonv',  # 替换为你的数据库用户名
            password='rpa@2025',  # 替换为你的数据库密码
            database='zmonv_rpa'  # 替换为你的数据库名
        )
        return conn
    except Error as e:
        print(f"数据库连接错误: {e}")
        return None


@tool()
def get_all_tables():
    """获取数据库中所有表的名称"""
    conn = create_db_connection()
    if not conn:
        return {"error": "无法连接数据库"}

    try:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        return {"tables": tables}
    except Error as e:
        return {"error": str(e)}
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

@tool()
def get_query_data(sql: str):
    """执行sql查询语句"""
    conn = create_db_connection()
    if not conn:
        return {"error": "无法连接数据库"}

    try:
        cursor = conn.cursor(dictionary=True)
        # 验证SQL语句类型，只允许查询或设置变量
        sql_clean = sql.strip().lower()
        allowed_prefixes = ['select', 'show', 'describe', 'explain', 'set']
        if not any(sql_clean.startswith(prefix) for prefix in allowed_prefixes):
            return {"error": "不允许执行该类型的SQL语句，仅支持查询或设置变量操作"}
        cursor.execute(sql)
        result = cursor.fetchall()
        query_data = {"data": result}
        return json.dumps(query_data, cls=DBEncoder)
    except Error as e:
        return {"error": str(e)}
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


@tool()
def get_table_schema(table_name: str):
    """获取表的创建信息"""
    conn = create_db_connection()
    if not conn:
        return {"error": "无法连接数据库"}

    try:
        cursor = conn.cursor()
        cursor.execute(f"SHOW CREATE TABLE {table_name}")
        result = cursor.fetchone()
        return {"schema": result[1]}  # 返回CREATE TABLE语句
    except Error as e:
        return {"error": str(e)}
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


load_dotenv()

if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_API_BASE"):
    raise ValueError("OPENAI_API_KEY and OPENAI_API_BASE must be set")

print(f"模型Base URL: {os.getenv("OPENAI_API_BASE")}")
CHAT_MODEL = os.getenv("CHAT_MODEL")
llm = ChatOpenAI(model=CHAT_MODEL, api_key=SecretStr(os.getenv("OPENAI_API_KEY")),base_url=os.getenv("OPENAI_API_BASE"))

config = {"configurable": {"thread_id": "1"}}

checkpointer = InMemorySaver()

agent = create_react_agent(model=llm,tools=[get_table_schema,get_all_tables,get_query_data],checkpointer=checkpointer)


class DBEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super(DBEncoder, self).default(obj)

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


    with st.chat_message("assistant"):

        full_response = ""
        history_components = []
        # 使用stream方法替代invoke，获取流式输出
        for chunk in agent.stream({"messages": st.session_state.messages},config=config):
            # 每个chunk包含当前步骤的结果
            print("\n===== 中间结果 =====")
            for key, value in chunk.items():
                print(f"{key}: {value}")
                if key == "tools":
                    messages = value["messages"]
                    for message in messages:
                        tool_name = message.name
                        content = json.loads(message.content)
                        st.markdown(f"### 调用工具：{tool_name}")
                        if tool_name == "get_all_tables":
                            if "error" in content:
                                st.error(f"错误: {content['error']}")
                                history_components.append({"type": "error", "content": content['error']})
                            else:
                                st.write("数据库表列表:")
                                history_components.append({"type": "text", "content": "数据库表列表:"})
                                st.code("\n".join(content["tables"]))
                                history_components.append({"type": "code", "content": content['tables']})
                        elif tool_name == "get_table_schema":
                            if "error" in content:
                                st.error(f"获取结构错误: {content['error']}")
                                history_components.append({"type": "error", "content": content['error']})
                            else:
                                st.code(content["schema"])
                                history_components.append({"type": "code", "content": content["schema"]})
                        elif tool_name == "get_query_data":
                            if "error" in content:
                                st.error(f"查询错误: {content['error']}")
                                history_components.append({"type": "error", "content": content['error']})
                            else:
                                st.subheader("查询结果:")
                                history_components.append({"type": "text", "content": "查询结果:"})
                                st.dataframe(content["data"])
                                history_components.append({"type": "dataframe", "content": content["data"]})
                elif key == "agent":
                    messages = value["messages"]
                    for message in messages:
                        tool_calls = message.additional_kwargs.get("tool_calls")
                        if tool_calls is None:
                            st.markdown(message.content)
                        else:
                            for tool_call in tool_calls:
                                func = tool_call["function"]
                                name = func["name"]
                                arguments = json.loads(func["arguments"])
                                if name == "get_query_data":
                                    st.write("执行SQL:")
                                    st.code(arguments["sql"])
                                elif name == "get_table_schema":
                                    st.write(arguments["table_name"])
                else:
                    print(f"其他键: {key}")

        st.session_state.chat_history.append({"role": "assistant", "components": history_components})





