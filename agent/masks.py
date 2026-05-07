import re
from typing import Dict, List


CHAT_BLOCK_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n?", re.DOTALL)


def classify_rendered_chat_spans(rendered_text: str) -> List[Dict[str, object]]:
    spans = []
    for match in CHAT_BLOCK_RE.finditer(rendered_text):
        role = match.group(1)
        content = match.group(2)
        is_tool_response = "<tool_response>" in content and "</tool_response>" in content
        if role == "assistant":
            label = "assistant_action"
            trainable = True
        elif is_tool_response:
            label = "tool_observation"
            trainable = False
        else:
            label = f"{role}_context"
            trainable = False
        spans.append(
            {
                "role": role,
                "label": label,
                "trainable": trainable,
                "start": match.start(),
                "end": match.end(),
                "content": content,
            }
        )
    return spans


def build_rendered_char_action_mask(rendered_text: str) -> List[int]:
    mask = [0] * len(rendered_text)
    for span in classify_rendered_chat_spans(rendered_text):
        if span["trainable"]:
            for i in range(int(span["start"]), int(span["end"])):
                mask[i] = 1
    return mask

