from LLMJudge import Judge
from openai import OpenAI
import os

def PrometheusJudge():
    judge = Judge()
    judge.load_model("prometheus-eval/prometheus-7b-v2.0","prometheus-eval/prometheus-7b-v2.0")  # Example model, replace with actual model name

    input_text = "What is the capital of France?"

    output_text = judge.judge(input_text)
    print(f"Input: {input_text}\nOutput: {output_text}")

def GPT4Judge():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
   # client = OpenAI(api_key="YOUR_API_KEY")
    instructions = """You are an expert in programming languages. Your task is to provide clear and concise answers to programming-related questions."""

    response = client.responses.create(
        model="gpt-4.1", #  "o4-mini"
        instructions=instructions,
        input="How would I declare a variable for a last name?",
    )

    print(response.output_text)
