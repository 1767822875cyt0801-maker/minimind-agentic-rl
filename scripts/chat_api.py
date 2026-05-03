#==依赖导入与客户端初始化==
import os
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

#==会话状态初始化==
stream = True
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
history_messages_num = 0  

#==交互主循环==
#持续从命令行读取用户问题，并先把当前问题放入对话历史中
while True:
    query = input('[Q]: ')
    conversation_history.append({"role": "user", "content": query})
    #请求构造和发送
    #把模型名、历史消息、采样参数、流式开关、额外模板参数一起发给后端
    #response 是一个完整的响应对象，里面通常有id，choices，usage，model，usage，message.content
    response = client.chat.completions.create(
        model="minimind-local:latest",
        messages=conversation_history[-(history_messages_num or 1):],
        stream=stream,
        temperature=0.8,
        max_tokens=2048,
        top_p=0.8,
        #这套 MiniMind 本地服务支持 reasoning / thinking 模式
        extra_body={"chat_template_kwargs": {"open_thinking": True}, "reasoning_effort": "medium"} # 思考开关
    )
    #响应处理
    #分为两条路径，一条是流式，另一条是单次响应
    if not stream:
        assistant_res = response.choices[0].message.content
        print('[A]: ', assistant_res)
    else:
        #显示 reasoning，但不把 reasoning 记入会话历史
        print('[A]: ', end='', flush=True)
        assistant_res = ''
        for chunk in response:
            delta = chunk.choices[0].delta
            r = getattr(delta, 'reasoning_content', None) or ""
            c = delta.content or ""
            if r:
                print(f'\033[90m{r}\033[0m', end="", flush=True)
            if c:
                print(c, end="", flush=True)
            assistant_res += c
    #历史更新与收尾
    #把模型最终回答加入会话历史，然后输出空行，让下一轮交互更清晰
    conversation_history.append({"role": "assistant", "content": assistant_res})
    print('\n\n')