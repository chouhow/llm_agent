import re
from pathlib import Path
from typing import Iterator, Dict, Optional


class SQLKnowledgeParser:
    """
    解析包含SQL知识的Markdown文件，提供迭代器返回二级标题相关内容
    正确提取```sql和```之间的所有SQL内容（包括多行）
    """

    def __init__(self, file_path: str):
        """
        初始化解析器

        Args:
            file_path: Markdown文件路径
        """
        self.file_path = Path(file_path)
        self.content = self._read_file()
        self._current_position = 0

    def _read_file(self) -> str:
        """读取Markdown文件内容"""
        try:
            return self.file_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {self.file_path} 不存在")
        except Exception as e:
            raise Exception(f"读取文件时出错: {str(e)}")

    def __iter__(self) -> Iterator[Dict[str, Optional[str]]]:
        """迭代器协议实现"""
        # 重置位置
        self._current_position = 0
        current_category = None

        # 正则表达式模式
        # 匹配一级标题 (类别)
        h1_pattern = re.compile(r'^#\s+(.*?)$', re.MULTILINE)
        # 匹配二级标题 (问题)
        h2_pattern = re.compile(r'^##\s+(.*?)$', re.MULTILINE)
        # 匹配示例SQL部分，专门处理```sql代码块
        sql_pattern = re.compile(
            r'^###\s+示例sql\s*```sql\s*(.*?)\s*```',
            re.DOTALL | re.MULTILINE
        )
        # 匹配业务背景部分
        background_pattern = re.compile(
            r'^###\s+问题分析\s*(.*?)(?=^#{1,2}\s+|$)',
            re.DOTALL | re.MULTILINE
        )

        while True:
            # 查找下一个一级或二级标题
            h1_match = h1_pattern.search(self.content, self._current_position)
            h2_match = h2_pattern.search(self.content, self._current_position)

            # 没有更多标题，退出循环
            if not h1_match and not h2_match:
                break

            # 确定下一个要处理的标题
            if h1_match and (not h2_match or h1_match.start() < h2_match.start()):
                # 处理一级标题（类别）
                current_category = h1_match.group(1).strip()
                self._current_position = h1_match.end()
            else:
                # 处理二级标题（问题）
                question = h2_match.group(1).strip()
                self._current_position = h2_match.end()

                # 提取当前二级标题下的内容，直到下一个标题
                end_pos = len(self.content)
                next_h1 = h1_pattern.search(self.content, self._current_position)
                next_h2 = h2_pattern.search(self.content, self._current_position)

                if next_h1 and (not next_h2 or next_h1.start() < next_h2.start()):
                    end_pos = next_h1.start()
                elif next_h2:
                    end_pos = next_h2.start()

                section_content = self.content[self._current_position:end_pos]

                # 提取示例SQL（专门处理```sql代码块）
                sql_match = sql_pattern.search(section_content)
                sql = sql_match.group(1).strip() if sql_match else None

                # 提取业务背景
                background_match = background_pattern.search(section_content)
                background = background_match.group(1).strip() if background_match else None

                # 更新当前位置
                self._current_position = end_pos

                # 返回提取的信息
                yield {
                    'category': current_category,
                    'question': question,
                    'sql': sql,
                    'background': background
                }


# 使用示例
if __name__ == "__main__":
    try:
        parser = SQLKnowledgeParser("../sql_gen_questions.md")

        for item in parser:
            print(f"类别: {item['category']}")
            print(f"问题: {item['question']}")
            print(f"SQL示例:\n{item['sql']}\n")
            print(f"问题背景:\n{item['background']}\n")
            print("-" * 80)
    except Exception as e:
        print(f"发生错误: {str(e)}")
