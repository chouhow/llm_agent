import json

from langchain_core.messages import ToolMessage

from app.json_helper import DBEncoder
from db_tools import get_query_data, get_all_tables, get_table_schema



def call_tool(tool_call,tools_by_name):
    function_call = tool_call.get("function")
    arguments_string = function_call["arguments"]
    # 使用json模块解析参数字符串
    arguments = json.loads(arguments_string)
    # 获取函数实体
    function = tools_by_name[function_call["name"]]
    # 如果入参为空，则直接调用函数
    if arguments == {}:
        tool_result = function()
    # 否则，传入参数后调用函数
    else:
        tool_result = function(**arguments)
    content = json.dumps(tool_result, cls=DBEncoder)
    return content,function_call["name"]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self) -> None:
        self.tools_by_name = {
            "get_query_data": get_query_data,
            "get_all_tables": get_all_tables,
            "get_table_schema": get_table_schema
        }

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.get("tool_calls", []):

            content ,name= call_tool(tool_call,self.tools_by_name)
            outputs.append(
                {"role": "tool", "content": content, "name": name,
                 "tool_call_id": tool_call["id"]}
            )
        return {"messages": outputs}


class ManualToolNode:
    """A node that runs the tools manually specified in the input message."""

    def __init__(self, ) -> None:
        self.tools_by_name = {
            "get_query_data": get_query_data,
            "get_all_tables": get_all_tables,
            "get_table_schema": get_table_schema
        }

    def __call__(self, inputs: dict):
        if pre_calls := inputs.get("pre_call_tolls", []):
            outputs = []
            for tool_call in pre_calls:
                content, name = call_tool(tool_call, self.tools_by_name)
                outputs.append(
                    {"role": "tool", "content": content, "name": name,
                     "tool_call_id": tool_call["id"]}
                )
            return {"messages": outputs}
        else:
            return {"pre_call_tolls": []}
