import re
from datetime import datetime
from pathlib import Path
from time import sleep

import streamlit as st

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.tools import tool

from langchain_openai.chat_models import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store
from pydantic import SecretStr
from langchain_community.chat_models import ChatTongyi
from langgraph.prebuilt import create_react_agent
from mysql.connector import Error
import mysql.connector
from pymilvus import MilvusClient

import db_tools
from prompts.PromptManager import PromptManager
from rag.milvus_helper import search_questions
from tools import print_langgraph_result


# project_root = Path(__file__).resolve().parent.parent
# env_file_path = project_root / '.env.27'
# print(env_file_path)
# load_dotenv(dotenv_path=env_file_path)

def create_db_connection():
    """创建MySQL数据库连接"""
    try:
        # conn = mysql.connector.connect(
        #     host='192.168.100.27',
        #     user='zmonv',  # 替换为你的数据库用户名
        #     password='rpa@2025',  # 替换为你的数据库密码
        #     database='zmonv_rpa'  # 替换为你的数据库名
        # )
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',  # 替换为你的数据库用户名
            password='123456',  # 替换为你的数据库密码
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
        cursor.execute("SHOW TABLE STATUS;")
        tables = [{"Name": table[0], "Comment": table[-1]} for table in cursor.fetchall()]
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
        # 1. 去除注释行
        # 移除单行注释 (-- 注释)
        cleaned_sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        # 移除多行注释 (/* 注释 */)
        cleaned_sql = re.sub(r'/\*.*?\*/', '', cleaned_sql, flags=re.DOTALL)
        # 2. 按分号分割语句
        statements = [stmt.strip() for stmt in cleaned_sql.split(';') if stmt.strip()]
        # 3. 检查语句类型
        allowed_keywords = {'select', 'show', 'describe', 'explain', 'set'}
        for stmt in statements:
            # 获取语句的第一个单词（转换为小写）
            words = stmt.split()
            if not words:
                continue
            first_word = words[0].lower()
            # 检查是否在允许的关键词中
            if first_word not in allowed_keywords:
                return {"error": "不允许执行该类型的SQL语句，仅支持查询或设置变量操作"}
        # 4. 执行语句，直到找到第一个有结果的查询

        for stmt in statements:
            cursor.execute(stmt)
            # 如果有返回结果，立即返回
            if cursor.description:
                # 获取所有行
                result = cursor.fetchall()
                return {"data": result}
            else:
                print(f"语句执行成功，但无返回结果: {stmt}")
                continue
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


@tool()
def get_current_time():
    """获取当前时间，格式为YYYY年MM月DD日 HH时MM分SS秒"""
    current_time = datetime.now()
    return current_time.strftime("%Y年%m月%d日 %H时%M分%S秒")


tools = [get_table_schema, get_all_tables, get_query_data, get_current_time]

load_dotenv()

if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_API_BASE"):
    raise ValueError("OPENAI_API_KEY and OPENAI_API_BASE must be set")

print(f"模型Base URL: {os.getenv("OPENAI_API_BASE")}")
CHAT_MODEL = os.getenv("CHAT_MODEL")
# 阿里百炼
llm = ChatOpenAI(model="qwen3-32b", api_key=SecretStr(os.getenv("DASHSCOPE_API_KEY")),
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
# 27服务器
# llm = ChatOpenAI(model=os.getenv("SERVER27_CHAT_MODEL"), api_key=SecretStr("no_need"),base_url=os.getenv("SERVER27_API_BASE"))
# 硅基流动
# llm = ChatOpenAI(model=os.getenv("SILICONFLOW_CHAT_MODEL"), api_key=SecretStr(os.getenv("SILICONFLOW_API_KEY")),base_url=os.getenv("SILICONFLOW_API_BASE"))

config = {"configurable": {"thread_id": "1"}}

checkpointer = InMemorySaver()

# agent = create_react_agent(model=llm,tools=tools,checkpointer=checkpointer)

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import List, Dict, Any


def custom_append_only(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    自定义添加方法，只将新消息追加到状态中
    注意：left 参数包含当前状态中的所有消息
          right 参数包含要添加的新消息
    """
    # 如果 right 是单个字典，则转换为列表
    if not isinstance(right, list):
        right = [right]

    return left + right


class State(TypedDict):
    messages: Annotated[list, custom_append_only]
    pre_call_tolls: list


pre_call_tolls = []
graph_builder = StateGraph(State)
llm_with_tools = llm.bind_tools(tools)


def llm_call(state: State):
    print(state["messages"], flush=True)
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


from openai import OpenAI
import os


def get_message_role(msg_type: str) -> str:
    if msg_type == "human":
        return "user"
    elif msg_type == "ai":
        return "assistant"
    elif msg_type == "system":
        return "system"
    else:
        return msg_type


def openai_llm_call(state: State):
    client = OpenAI(
        # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # dict_messages = []
    # for msg in state["messages"]:
    #     role =get_message_role(msg.type)
    #     content = msg.content
    #     dict_msg={
    #         "role":role,
    #         "content":content
    #     }
    #     if hasattr(msg,"tool_call"):
    #         dict_msg["tool_calls"] = msg.tool_call
    #     dict_messages.append(dict_msg)

    # dict_messages = [{"role": get_message_role(msg.type), "content": str(msg.content)} for msg in state["messages"] ]
    # print(dict_messages)
    dict_messages = state["messages"]
    completion = client.chat.completions.create(
        # model="qwen3-32b",
        model="qwen3-235b-a22b",
        messages=dict_messages,
        parallel_tool_calls=True,
        tools=db_tools.tools,
        stream=True,
    )
    reasoning_content = ""  # 完整思考过程
    answer_content = ""  # 完整回复
    tool_info = []  # 存储工具调用信息
    for chunk in completion:
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
            continue

        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            print(delta.content, end="", flush=True)
            answer_content += delta.content
        # 只收集思考内容
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            print(delta.reasoning_content, end="", flush=True)
            reasoning_content += delta.reasoning_content
        # 处理工具调用信息（支持并行工具调用）
        if delta.tool_calls is not None:
            for tool_call in delta.tool_calls:
                index = tool_call.index  # 工具调用索引，用于并行调用

                # 动态扩展工具信息存储列表
                while len(tool_info) <= index:
                    tool_info.append({})

                # 收集工具调用ID（用于后续函数调用）
                if tool_call.id:
                    tool_info[index]['id'] = tool_info[index].get('id', '') + tool_call.id

                if tool_call.function and tool_info[index].get("function") is None:
                    tool_info[index]["function"] = {}
                # 收集函数名称（用于后续路由到具体函数）
                if tool_call.function and tool_call.function.name:
                    tool_info[index]["function"]['name'] = tool_info[index]["function"].get('name',
                                                                                            '') + tool_call.function.name

                # 收集函数参数（JSON字符串格式，需要后续解析）
                if tool_call.function and tool_call.function.arguments:
                    tool_info[index]["function"]['arguments'] = tool_info[index]["function"].get('arguments',
                                                                                                 '') + tool_call.function.arguments
    print(f"\n" + "=" * 19 + "工具调用信息" + "=" * 19)
    if not tool_info:
        print("没有工具调用")
    else:
        print(tool_info)

    return {"messages": [{"role": "assistant", "content": answer_content, "tool_calls": tool_info}]}


def route_tools(
        state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    print("route_tools ai_message", ai_message)
    if "tool_calls" in ai_message and len(ai_message["tool_calls"]) > 0:
        print("to tools")
        return "tools"
    print("to end")
    return END


from app_tool import BasicToolNode, ManualToolNode

graph_builder.add_node("llm_call", openai_llm_call)
tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
pre_tools_node = ManualToolNode(tools=tools)
graph_builder.add_node("pre_tools", pre_tools_node)

graph_builder.set_entry_point("pre_tools")
# graph_builder.set_entry_point("llm_call")
graph_builder.add_edge("pre_tools", "llm_call")
graph_builder.add_conditional_edges("llm_call", route_tools, {"tools": "tools", END: END}, )
graph_builder.add_edge("tools", "llm_call")
graph = graph_builder.compile()

# 页面配置
st.set_page_config(
    page_title="LLM对话助手",
    page_icon="💬",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_vector_store(URI="http://192.168.100.27:19530"):
    client = MilvusClient(uri=URI)
    return client


promptmanager = PromptManager("../background_prompts.md")


def build_prompt(question: str):
    prompt = """
## 角色
你是公司内部的财务助手，精通财务报表和发票税务知识，同时擅长使用SQL语句。
## 任务
根据用户的问题，生成并执行相应的mysql语句。 生成sql语句后，需要调用工具执行sql语句。   
查询完sql后，使用一两行回复用户执行完成即可，不要在回复中列出详细记录。
## 注意事项
注意选择合适的工具来完成任务，包括执行sql查询，获取表结构等。   
根据用户的问题，简洁地回答问题，避免无关输出；
## 输出格式
输出给用户的回答，请遵循markdown语法。 
"""

    entities = search_questions(question)

    hit_question = [entity['question_text'] for entity in entities]

    print("hit_question:", hit_question)
    entity = entities[0]
    category = entity['category']
    print("category:", category)
    category_prompt = promptmanager.get_prompt(category)
    prompt += ("\n## 业务背景\n")
    prompt += category_prompt

    examples = [
        {"question": entity['question_text'], "explain": entity['question_context'], "sql": entity['example_sql']} for
        entity in entities]
    from langchain.prompts import PromptTemplate, FewShotPromptTemplate
    example_template = """
    示例问题: {question}
    问题分析: {explain}
    示例SQL: {sql}
    """
    example_prompt = PromptTemplate(
        input_variables=["question", "explain", "sql"],
        template=example_template,
    )
    prefix = """
## 示例问题和SQL语句
"""
    examples_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix="请分析业务背景和示例问题，选择和用户问题相关的示例，生成满足用户需要的sql，然后调用工具执行。如果示例问题没有和用户问题相关的，直接回复根据现有知识，无法回答！。",
        example_separator="\n"  # 示例之间的分隔符
    )

    prompt += examples_prompt.format(input=question)

    # if category == "科目余额表":
    #     tool_call = {
    #         "name": "get_table_schema",
    #         "args": {
    #             "table_name": "jd_account_balance_table"
    #         }
    #     }
    #     pre_call_tolls.append(tool_call)
    #     print("pre_call_tolls",pre_call_tolls)

    # prompt += f"示例问题：{entity['question_text']}\n"
    # prompt += f"对应SQL语句：{entity['example_sql']}\n\n"
    print("prompt:", prompt)
    return prompt


# 清除对话函数
def clear_chat():
    # 删除session_state中的messages
    if "messages" in st.session_state:
        del st.session_state.messages
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
if user_input := st.chat_input("请输入您的问题..."):

    prompt = build_prompt(user_input)
    system_prompt = {"role": "system", "content": prompt}
    if "messages" not in st.session_state:
        st.session_state.messages = [system_prompt]

    # 将用户输入添加到消息列表
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 显示用户输入
    with st.chat_message("user"):
        st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "components": [{"type": "text", "content": user_input}]})

    with st.chat_message("assistant"):

        full_response = ""
        history_components = []
        # 使用stream方法替代invoke，获取流式输出
        for chunk in graph.stream({"messages": st.session_state.messages, "pre_call_tolls": pre_call_tolls},
                                  config=config):
            # 每个chunk包含当前步骤的结果
            print("\n===== 中间结果 =====")
            for key, value in chunk.items():
                print(f"{key}: {value}")
                if key == "tools":
                    messages = value["messages"]
                    for message in messages:
                        tool_name = message.get("name")
                        content = json.loads(message.get("content"))
                        st.markdown(f"### 调用工具：{tool_name}")
                        if tool_name == "get_all_tables":
                            if "error" in content:
                                st.error(f"错误: {content['error']}")
                                history_components.append({"type": "error", "content": content['error']})
                            else:
                                st.write("数据库表列表:")
                                history_components.append({"type": "text", "content": "数据库表列表:"})
                                st.dataframe(content["tables"])
                                history_components.append({"type": "dataframe", "content": content['tables']})
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
                elif key == "llm_call":
                    messages = value["messages"]
                    for message in messages:
                        tool_calls = message.get("tool_calls")
                        if tool_calls is None or len(tool_calls) == 0:
                            st.markdown(message.get("content"))
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
