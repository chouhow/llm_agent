from pathlib import Path


class PromptManager:
    def __init__(self, markdown_file_path=None):
        """
        初始化提示词管理器

        参数:
            markdown_file_path (str, optional): Markdown文件路径
        """
        self.prompt_library = {}
        self.default_prompt = "请根据以下内容提供帮助："

        if markdown_file_path:
            self.load_from_markdown(markdown_file_path)

    def load_from_markdown(self, file_path):
        """
        从Markdown文件中读取提示词（按行读取，不使用正则表达式）

        参数:
            file_path (str): Markdown文件路径

        返回:
            bool: 是否成功加载
        """
        try:
            # 检查文件是否存在
            path = Path(file_path)
            if not path.exists():
                print(f"错误: 文件 '{file_path}' 不存在")
                return False

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            current_prompt_type = None
            in_template_section = False
            template_content = []

            for line in lines:
                stripped_line = line.strip()

                # 检查是否是三级标题（提示词类型）
                if line.startswith("### ") :
                    if current_prompt_type:
                        self.prompt_library[current_prompt_type] = " ".join(template_content).strip()
                    current_prompt_type = stripped_line[4:].strip()
                    template_content = []
                else:
                    if len(stripped_line)>0:
                        template_content.append(line)
            if current_prompt_type not in self.prompt_library:
                self.prompt_library[current_prompt_type] = " ".join(template_content).strip()
            return True

        except Exception as e:
            print(f"读取文件时出错: {e}")
            return False

    def get_prompt(self, prompt_type, user_input=None):
        """
        根据用户传入的提示词类型返回对应的提示词

        参数:
            prompt_type (str): 提示词类型
            user_input (str, optional): 用户输入的内容

        返回:
            str: 完整的提示词
        """
        # 获取对应的提示词，如果没有找到则使用默认提示词
        base_prompt = self.prompt_library.get(prompt_type, self.default_prompt)

        # 如果用户提供了输入内容，则将其附加到提示词后
        if user_input:
            return f"{base_prompt} {user_input}"
        else:
            return base_prompt

    def add_prompt(self, prompt_type, prompt_template):
        """
        添加新的提示词类型到库中

        参数:
            prompt_type (str): 新的提示词类型名称
            prompt_template (str): 对应的提示词模板
        """
        self.prompt_library[prompt_type] = prompt_template

    def list_prompt_types(self):
        """
        列出所有可用的提示词类型

        返回:
            list: 所有可用的提示词类型列表
        """
        return list(self.prompt_library.keys())

    def remove_prompt_type(self, prompt_type):
        """
        从库中移除提示词类型

        参数:
            prompt_type (str): 要移除的提示词类型

        返回:
            bool: 是否成功移除
        """
        if prompt_type in self.prompt_library:
            del self.prompt_library[prompt_type]
            return True
        return False

    def export_to_markdown(self, file_path):
        """
        将当前提示词库导出到Markdown文件

        参数:
            file_path (str): 导出的文件路径
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                for prompt_type, prompt_template in self.prompt_library.items():
                    file.write(f"### {prompt_type}\n\n")
                    file.write(f"#### prompt_template\n\n")
                    file.write(f"{prompt_template}\n\n")
                    file.write("---\n\n")

            print(f"提示词库已成功导出到 {file_path}")
            return True
        except Exception as e:
            print(f"导出文件时出错: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    # 创建提示词管理器实例并加载Markdown文件
    manager = PromptManager("../background_prompts.md")

    # 获取写作提示词
    balance_prompt = manager.get_prompt("科目余额表")
    print("科目余额表:", balance_prompt)

    # 列出所有可用的提示词类型
    print("可用提示词类型:", manager.list_prompt_types())

