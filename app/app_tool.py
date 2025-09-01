import json

from langchain_core.messages import ToolMessage

from app.json_helper import DBEncoder


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            print("tool_result",tool_result)
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result, cls=DBEncoder),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


class ManualToolNode:
    """A node that runs the tools manually specified in the input message."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if pre_calls := inputs.get("pre_call_tolls", []):
            outputs = []
            for tool_call in pre_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                print(tool_result)
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["name"],
                    )
                )
            return {"messages": outputs}
        else:
            return {"pre_call_tolls": []}
