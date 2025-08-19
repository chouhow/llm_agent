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
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
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
        # 验证SQL语句类型，只允许查询或设置变量
        sql_clean = sql.strip().lower()
        allowed_prefixes = ['select', 'show', 'describe', 'explain', 'set']
        if not any(sql_clean.startswith(prefix) for prefix in allowed_prefixes):
            return {"error": "不允许执行该类型的SQL语句，仅支持查询或设置变量操作"}
        cursor.execute(sql)
        result = cursor.fetchall()
        return {"data": result}
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