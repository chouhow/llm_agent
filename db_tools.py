import re

import mysql.connector
from mysql.connector import Error


# ===================== 数据库工具 =====================
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



def get_all_tables():
    """获取数据库中所有表的名称"""
    conn = create_db_connection()
    if not conn:
        return {"error": "无法连接数据库"}

    try:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLE STATUS;")
        tables = [{"Name":table[0],"Comment":table[-1]} for table in cursor.fetchall()]
        return {"tables": tables}
    except Error as e:
        return {"error": str(e)}
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


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
                total_rows = len(result)
                # 限制返回结果为最多10条
                if total_rows > 10:
                    result = result[:10]  # 只保留前10条
                    return {"data": result, "rows": len(result), "total_rows": total_rows, "message": "结果超过10条，仅显示前10条"}
                else:
                    return {"data": result, "rows": total_rows}
            else:
                print(f"语句执行成功，但无返回结果: {stmt}")
                continue
    except Error as e:
        return {"error": str(e)}
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()



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
            "description": "执行SQL查询语句,可以有多个语句,支持select,show,describe,explain,set。",
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

if __name__ == '__main__':
   # print(get_all_tables())

    sql = """
    set @last_month=(SELECT file_date from jd_account_balance_table where subject_id = '1221' ORDER BY file_date desc LIMIT 1);
    -- 根据科目代码和日期查出其他应收款明细
    SELECT file_date as 日期,subject_name as 科目名称,functional_currency as 币别,initial_debit_balance as 期初借方余额,initial_credit_balance as 
    期初贷方余额,current_debit_amount as 本期借方发生额,current_credit_amount as 本期贷方发生额,accumulated_debit_year as 本年借方累计,accumulated_credit_year as 
    本年贷方累计,ending_debit_balances as 期末借方余额,ending_credit_balances as 期末贷方余额 from jd_account_balance_table where subject_id like '1221%' AND file_date=@last_month;
    
    """
    print(get_query_data(sql))
