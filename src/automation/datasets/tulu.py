
def make_tulu_prompt(sample):
    msgs = []
    for m in sample["messages"]:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        msgs.append(f"{role.upper()}: {content}")
    prompt = "\n".join(msgs)

    return prompt
