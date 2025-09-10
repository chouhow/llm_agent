import unittest
from db_tools import create_db_connection, get_all_tables, get_query_data, get_table_schema

class TestDBTools(unittest.TestCase):
    # 测试数据库配置（建议使用专用测试库）
    TEST_DB_CONFIG = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': '123456',
        'database': 'zmonv_rpa'  # 建议使用测试专用数据库
    }
    TEST_TABLE = 'corp_info'

    def test_create_db_connection(self):
        conn = create_db_connection()
        self.assertIsNotNone(conn)
        self.assertTrue(conn.is_connected())
        conn.close()

    def test_get_all_tables(self):
        result = get_all_tables()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('tables', result)
        self.assertIn(self.TEST_TABLE, result['tables'])
        # self.assertIsInstance(result['tables'], list)


    def test_get_query_data(self):
        result = get_query_data(f"SELECT * FROM {self.TEST_TABLE}")
        self.assertIn('data', result)
        self.assertTrue(len(result['data'])>0)



        result = get_query_data(f"DELETE FROM {self.TEST_TABLE}")
        self.assertIn('error', result)
        self.assertIn('不允许执行该类型的SQL语句', result['error'])

    def test_get_table_schema(self):
        result = get_table_schema(self.TEST_TABLE)
        self.assertIn('schema', result)


if __name__ == '__main__':
    unittest.main()