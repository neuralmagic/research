
def make_tulu_prompt(sample):
    msgs = []
    for m in sample["messages"]:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        msgs.append(f"{role.upper()}: {content}")
    joined = "\n".join(msgs)
    prompt = f"### Conversation:\n{joined}\n\n### Response:\n"

    return prompt
