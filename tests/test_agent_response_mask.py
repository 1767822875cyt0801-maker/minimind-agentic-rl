from agent.masks import classify_rendered_chat_spans


def test_rendered_chat_mask_marks_only_assistant_actions():
    rendered = (
        '<|im_start|>system\n'
        '# Tools\n<tools>{"name":"calculate_math"}</tools><|im_end|>\n'
        '<|im_start|>user\n'
        '帮我计算 2+2<|im_end|>\n'
        '<|im_start|>assistant\n'
        '<tool_call>{"name":"calculate_math","arguments":{"expression":"2+2"}}</tool_call><|im_end|>\n'
        '<|im_start|>user\n'
        '<tool_response>\n{"result":4}\n</tool_response><|im_end|>\n'
        '<|im_start|>assistant\n'
        '2+2 = 4<|im_end|>\n'
    )
    spans = classify_rendered_chat_spans(rendered)
    labels = [span["label"] for span in spans]
    trainable = [span["trainable"] for span in spans]
    assert labels == [
        "system_context",
        "user_context",
        "assistant_action",
        "tool_observation",
        "assistant_action",
    ]
    assert trainable == [False, False, True, False, True]

