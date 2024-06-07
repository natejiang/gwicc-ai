import os
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

# 加载 .env 到环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

vn = MyVanna(config={'api_key': os.getenv("DASHSCOPE_API_KEY"), 'model': 'qwen-turbo', "base_url": os.getenv("OPENAI_BASE_URL")})

vn.connect_to_sqlite('.\database\database.sqlite')

from vanna.flask import VannaFlaskApp
VannaFlaskApp(vn).run()
