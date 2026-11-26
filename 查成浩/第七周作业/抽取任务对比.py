"""
抽取任务对比脚本（作业版）
用法: python extract_compare.py
依赖: pip install openai google-genai pydantic python-dotenv
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
from google import genai
from google.genai import types

load_dotenv()

# ---------- 1. 定义抽取目标 ----------
class PersonInfo(BaseModel):
    name: str = Field(description="人物姓名")
    email: str = Field(description="电子邮箱，没有则填空字符串")
    role: str = Field(description="职位或身份")


# ---------- 2. 两个抽取函数 ----------
def extract_openai(text: str) -> PersonInfo:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.beta.chat.completions.parse(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "你是信息抽取器，只提取文中明确出现的信息。"},
            {"role": "user", "content": text},
        ],
        response_format=PersonInfo,
    )
    return resp.choices[0].message.parsed


def extract_gemini(text: str) -> PersonInfo:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=text,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PersonInfo,
        ),
    )
    return PersonInfo.model_validate_json(resp.text)


# ---------- 3. 主流程：同一段文本，跑两个模型，对比 ----------
if __name__ == "__main__":
    text = "联系 John Smith，他是 Acme Corp 的技术总监，邮箱 john.smith@acme.com。"

    print("输入文本：", text, "\n")

    for name, func in [("OpenAI (gpt-4.1-mini)", extract_openai),
                       ("Gemini (2.5-pro)", extract_gemini)]:
        print(f"--- {name} ---")
        try:
            result = func(text)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print("出错：", e)
        print()
