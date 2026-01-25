from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# Тест MiMo
result = client.chat.completions.create(
    model="xiaomi/mimo-v2-flash:free",
    messages=[{"role": "user", "content": "Привет!"}]
)
print(result.choices[0].message.content)