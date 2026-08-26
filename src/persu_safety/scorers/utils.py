import re

from inspect_ai.model import ChatMessage


def parse_messages_into_transcript(messages: list[ChatMessage]) -> str:
    converted_messages: list[str] = []
    for m in messages:
        if m.role == "assistant":
            converted_messages.append("Persuader: " + str(m.content) + "\n")
        elif m.role == "user":
            converted_messages.append("Persuadee: " + str(m.content) + "\n")
    transcript: str = "\n".join(converted_messages)
    return transcript


def strip_to_json(text: str) -> str:
    """Convert model output to valid JSON string"""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text
