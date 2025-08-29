from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient, DataType

from rag.markdown_parser import SQLKnowledgeParser
from uuid import uuid4

URI = "http://192.168.100.27:19530"

embeddings = OpenAIEmbeddings(model="BAAI/bge-m3", base_url="https://api.siliconflow.cn/v1",
                              api_key="sk-mafsahwuuvvcxeycvpgjtzclsedmfconaxcnfodwtolvbdzx")

Questions_Collection = "sql_gen_questions"
def create_milvus_collection_question_sql():

    client = MilvusClient(uri=URI)

    schema = client.create_schema()

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="question_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="question_text", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="business_context", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="example_sql", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=32)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="question_vector", index_type="AUTOINDEX", metric_type="L2")


    client.create_collection(collection_name=Questions_Collection, schema=schema, index_params=index_params)


def insert_question_to_milvus():
    data = []
    try:
        parser = SQLKnowledgeParser("../sql_gen_questions.md")

        for item in parser:
            category = item["category"]
            question = item["question"]
            business_context = item["background"]
            example_sql = item["sql"]
            question_vector = embeddings.embed_query(question)
            obj= {
                "question_vector": question_vector,
                "question_text": question,
                "business_context": business_context,
                "example_sql": example_sql,
                "category": category
            }
            print(obj)
            data.append(obj)

    except Exception as e:
        print(f"发生错误: {str(e)}")
    client = MilvusClient(uri=URI)
    res = client.insert(
        collection_name=Questions_Collection,
        data=data
    )
    return res


def search_questions(question: str):
    query_vector = embeddings.embed_query(question)
    client = MilvusClient(uri=URI)
    res = client.search(
        collection_name=Questions_Collection,
        data=[query_vector],
        limit=3,
        output_fields=["question_text", "business_context", "example_sql"],
        search_params={"metric_type": "L2"}
    )
    entities = [ r["entity"] for r in res[0]]
    return entities

if __name__ == '__main__':
    # create_milvus_collection_question_sql()
    # insert_question_to_milvus()
    res = search_questions("广告")
    print(res)
