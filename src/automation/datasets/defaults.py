def make_default_prompt(sample):
    messages = [
        {
            "role": "user",
            "content": f"{json.dumps(sample)}",
        }
    ]

    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    return prompt

