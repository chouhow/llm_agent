import streamlit as st
import time

# 页面配置
st.set_page_config(
    page_title="LLM对话助手",
    page_icon="💬",
    layout="wide"
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# 模拟LLM回复函数（实际应用中替换为真实API调用）
def get_llm_response(user_input, chat_history):
    """
    模拟LLM生成回复的函数

    参数:
        user_input: 用户当前输入
        chat_history: 历史对话记录

    返回:
        模拟的LLM回复
    """
    # 这里只是模拟延迟和回复，实际应用中替换为真实的LLM API调用
    time.sleep(1)  # 模拟网络延迟

    # 简单的回复逻辑，实际应用中应使用真实的LLM
    return f"这是对您问题的回复：'{user_input}'。（这是模拟回复，实际应用中会调用真实的LLM）"


# 清除对话函数
def clear_chat():
    st.session_state.messages = []
    st.session_state.chat_history = []


# 页面标题和说明
st.title("💬 LLM对话助手")


# 显示清除对话按钮
st.sidebar.button("清除对话历史", on_click=clear_chat, type="primary")


# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 将用户输入添加到消息列表
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示用户输入
    with st.chat_message("user"):
        st.markdown(prompt)

    # 准备上下文（历史对话）
    context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])

    # 生成LLM回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 调用LLM获取回复（这里使用模拟函数）
        llm_response = get_llm_response(prompt, st.session_state.chat_history)


        message_placeholder.markdown(full_response)

    # 将AI回复添加到消息列表
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 更新对话历史（用于上下文管理）
    st.session_state.chat_history.append({"user": prompt, "assistant": full_response})
