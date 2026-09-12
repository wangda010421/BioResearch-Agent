import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
    )
