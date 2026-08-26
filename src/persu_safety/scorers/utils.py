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
