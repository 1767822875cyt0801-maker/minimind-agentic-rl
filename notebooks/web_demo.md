给 MiniMind 提供一个可直接运行的本地网页聊天界面
本质上是一个推理 + UI 编排器
- `web_demo.py`：**本地网页前端**，用户在浏览器里直接跟模型聊
- `chat_api.py`：**程序接口前端**，通过 OpenAI 风格 API 调模型

- **搭界面**  
    用 Streamlit 生成网页，包括标题、logo、侧边栏、聊天输入框、聊天记录区。
- **加载模型和 tokenizer**  
    通过 `AutoModelForCausalLM.from_pretrained(...)` 和 `AutoTokenizer.from_pretrained(...)` 从本地模型目录加载 MiniMind。
- **管理聊天状态**  
    用 `st.session_state` 保存当前会话中的用户消息、模型消息、配置参数、语言、工具勾选状态等。
- **把对话整理成模型能吃的 prompt**  
    通过 `tokenizer.apply_chat_template(...)` 把多轮消息、system prompt、thinking 开关、tools 定义拼装成最终输入文本。
- **执行流式生成**  
    使用 `TextIteratorStreamer` + `Thread(target=model.generate, ...)`，让模型一边生成，前端一边显示。
- **处理 thinking 和 tool call**  
    它不仅展示普通回答，还会解析 `<think>...</think>` 和 `<tool_call>...</tool_call>`，并在检测到工具调用时执行本地 Python 函数，再把工具结果继续喂回模型。