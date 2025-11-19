
def make_tulu_prompt(sample):
    messages = sample["messages"][:-1]
    return messages

"""
def make_tulu_prompt(sample):
    messages = []
    for m in sample["messages"]:
        role = m.get("role", "")
        content = m.get("content", "").strip()
        if role == "user":
            convo = {
                "role": role,
                "content": content,
            }
            messages.append(convo)

    return messages
"""
