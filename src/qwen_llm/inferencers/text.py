"""将累计模型输出拆分为累计思考与正文；保留不完整标签直至后续分块。"""

def _strip_leading_closers(text: str, final: bool) -> str:
    candidate = text
    removed = False
    marker = "</think>"
    while candidate.lstrip().startswith(marker):
        candidate = candidate.lstrip()[len(marker):]
        removed = True
    if not final and marker.startswith(candidate.lstrip()):
        return ""
    return candidate.lstrip() if removed else candidate


def split_thinking(text: str, thinking: bool, final: bool = False) -> tuple[str, str]:
    if not thinking:
        return "", _strip_leading_closers(text, final)
    if text.startswith("<think>"):
        text = text[7:]
    elif "<think>".startswith(text) and not final:
        return "", ""
    marker = "</think>"
    if marker in text:
        reasoning, answer = text.split(marker, 1)
        return reasoning, _strip_leading_closers(answer, final)
    if not final:
        for size in range(len(marker) - 1, 0, -1):
            if text.endswith(marker[:size]):
                return text[:-size], ""
    return text, ""
