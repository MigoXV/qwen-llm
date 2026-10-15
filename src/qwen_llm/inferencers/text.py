"""将累计模型输出拆分为累计思考与正文；保留不完整标签直至后续分块。"""


def split_thinking(text: str, thinking: bool, final: bool = False) -> tuple[str, str]:
    if not thinking:
        return "", text
    if text.startswith("<think>"):
        text = text[7:]
    elif "<think>".startswith(text) and not final:
        return "", ""
    marker = "</think>"
    if marker in text:
        reasoning, answer = text.split(marker, 1)
        return reasoning, answer
    if not final:
        for size in range(len(marker) - 1, 0, -1):
            if text.endswith(marker[:size]):
                return text[:-size], ""
    return text, ""
