import json
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from pymilvus import MilvusClient

import db_tools
from prompts.PromptManager import PromptManager
from rag.milvus_helper import search_questions

load_dotenv()

if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_API_BASE"):
    raise ValueError("OPENAI_API_KEY and OPENAI_API_BASE must be set")

config = {"configurable": {"thread_id": "1"}}

checkpointer = InMemorySaver()

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

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
    is_new_question: bool


pre_call_tolls = []
graph_builder = StateGraph(State)


# 添加新的判断节点函数
def check_question_relevance(state: State):
    # 获取历史消息和当前用户输入
    messages = state["messages"]
    if not messages:
        return {"is_new_question": True}
    print("1")

    # 提取最近的用户消息
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    print(user_messages)
    if not user_messages or len(user_messages) <= 1:
        return {"is_new_question": True}

    print("2")
    current_question = user_messages[-1]["content"]
    print("3")
    # 构建用于判断相关性的提示
    relevance_prompt = {
        "role": "system",
        "content": "你需要判断用户的当前问题是否与之前的对话相关。请输出'相关'或'不相关'，不要输出其他内容。"
    }
    # 准备发送给大模型的消息
    history_messages = messages[:-1]  # 除了当前用户消息外的所有历史消息
    check_relevant = {"role": "user", "content": f"历史对话: {history_messages}\n\n当前问题: {current_question}"}
    chat_messages = [relevance_prompt,check_relevant]
    for msg in chat_messages:
        print(msg)
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen3-235b-a22b",
            messages=chat_messages,
            max_tokens=10,
            stream=True,
        )

        result = process_streaming_response(completion)
        answer_content = result["answer_content"]
        print(f"问题相关性判断结果: {answer_content}")
        # 根据结果设置标识
        if "不相关" in answer_content:
            return {"is_new_question": True}
        else:
            return {"is_new_question": False}
    except Exception as e:
        print(f"判断问题相关性时出错: {e}")
        # 出错时默认视为相关问题
        return {"is_new_question": False}


# 添加一个处理新问题的节点
def handle_new_question(state: State):
    user_message = [msg for msg in state["messages"] if msg.get("role") == "user"][-1]
    # 清空消息
    state["messages"] = []
    question = user_message["content"]
    system_prompt = build_prompt(question)
    system_message = {"role": "system", "content": system_prompt}
    # 重新构建messages，只包含系统提示和当前用户输入
    new_messages = [system_message, user_message]
    return {"messages": new_messages}


def route_by_relevance(state: State):
    if state["is_new_question"]:
        return "handle_new_question"
    else:
        return "pre_tools"


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


def process_streaming_response(completion):
    """
    处理OpenAI API的流式输出响应

    参数:
        completion: OpenAI API返回的流式响应对象

    返回:
        dict: 包含answer_content, reasoning_content和tool_info的字典
    """
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
        if  hasattr(delta,"tool_calls") and delta.tool_calls is not None:
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

    return {
        "answer_content": answer_content,
        "reasoning_content": reasoning_content,
        "tool_info": tool_info
    }


def openai_llm_call(state: State):
    client = OpenAI(
        # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    dict_messages = state["messages"]
    print("openai_llm_call",dict_messages)
    completion = client.chat.completions.create(
        # model="qwen3-32b",
        model="qwen3-235b-a22b",
        messages=dict_messages,
        parallel_tool_calls=True,
        tools=db_tools.tools,
        stream=True,
    )
    result = process_streaming_response(completion)
    answer_content = result["answer_content"]
    tool_info = result["tool_info"]
    print(f"\n" + "=" * 19 + "工具调用信息" + "=" * 19)
    if not tool_info:
        print("没有工具调用")
    else:
        print(tool_info)
    # 如果tool_info为空就不要加上，否则报错：  Empty tool_calls is not supported in message
    if tool_info:
        return {"messages": [{"role": "assistant", "content": answer_content, "tool_calls": tool_info}]}
    else:
        return {"messages": [{"role": "assistant", "content": answer_content}]}


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

tool_node = BasicToolNode()
pre_tools_node = ManualToolNode()

graph_builder.add_node("check_relevance", check_question_relevance)
graph_builder.add_node("handle_new_question", handle_new_question)
graph_builder.add_node("llm_call", openai_llm_call)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("pre_tools", pre_tools_node)

graph_builder.set_entry_point("check_relevance")  # 入口改为判断相关性
graph_builder.add_conditional_edges(
    "check_relevance",
    route_by_relevance,
    {"handle_new_question": "handle_new_question", "pre_tools": "pre_tools"}
)
graph_builder.add_edge("handle_new_question", "pre_tools")
graph_builder.add_edge("pre_tools", "llm_call")
graph_builder.add_conditional_edges("llm_call", route_tools, {"tools": "tools", END: END})
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
if "messages" not in st.session_state:
    print("init session_state messages")
    st.session_state.messages = []


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
## 输出格式
输出给用户的回答，请遵循markdown语法,简要总结即可，避免无关输出，不用输出详细的sql语句，也不用输出查询结果。 
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
    print("prompt:", prompt)
    return prompt


# 清除对话函数
def clear_chat():
    # 删除session_state中的messages
    # if "messages" in st.session_state:
    #     del st.session_state.messages
    st.session_state.chat_history = []
    st.session_state.messages = []


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

    print("handle input messages ", [msg.get("role") for msg in st.session_state.messages])
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
            # 检查是否是新问题，如果是则清除聊天历史
            if "check_relevance" in chunk and chunk["check_relevance"].get("is_new_question"):
                print("clear chat ")
                clear_chat()

            # 每个chunk包含当前步骤的结果
            print("\n===== 中间结果 =====")
            for key, value in chunk.items():
                print(f"{key}: {value}")
                if "messages" in value:
                    print(f"add {key} messages:")
                    st.session_state.messages.extend(value["messages"])
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
