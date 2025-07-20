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

def test_rate_limits():
    import os
    import httpx
    import json

    client = httpx.Client(
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json"
        },
        timeout=30.0
    )

    def query_openai(messages, model="gpt-4.1", temperature=0.2):
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        response = client.post(url, json=payload)
        data = response.json()

        # Extract headers
        headers = response.headers
        rate_info = {
            "requests_left": headers.get("x-ratelimit-remaining-requests"),
            "tokens_left": headers.get("x-ratelimit-remaining-tokens"),
            "requests_reset": headers.get("x-ratelimit-reset-requests"),
            "tokens_reset": headers.get("x-ratelimit-reset-tokens"),
        }
        if 'choices' not in data:
            return data, rate_info
        return data["choices"][0]["message"]["content"], rate_info

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how to use the OpenAI API."}
    ]

    response_text, rate_info = query_openai(messages)
    print("Response:", response_text)
    print("Rate Info:", rate_info)

test_rate_limits()