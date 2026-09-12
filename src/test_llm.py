import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# 读取项目根目录中的 .env
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件")


# DeepSeek 提供 OpenAI-compatible API
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0,
    timeout=30,
    max_retries=1,
)


response = llm.invoke(
    "In one sentence, explain what CRISPR-Cas9 is."
)

print("LLM connection successful!")
print("\nModel response:")
print(response.content)
