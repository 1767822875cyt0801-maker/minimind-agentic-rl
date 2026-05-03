把已经训练好的 MiniMind 模型加载起来，然后做推理对话测试
完成下面这四件事
 - **解析命令行参数**  
    决定加载哪个模型、哪个权重、是否带 LoRA、是否用 MoE、生成参数是多少、是否保留历史对话等。
- **初始化 tokenizer 和 model**  
    支持两种加载方式：
    - 加载你自己这个项目里的 **原生 MiniMind 模型结构 + `.pth` 权重**
    - 或者直接加载 HuggingFace / Transformers 格式模型
- **构造输入 prompt**  
    根据当前使用的权重类型，决定输入是：
    - 预训练风格：`bos + prompt`
    - 对话风格：`apply_chat_template(...)`
- **调用 `model.generate()` 做生成**  
    然后实时打印输出、统计生成速度、更新多轮对话历史

#### 完整执行流程
##### 第一步：读参数
决定：
- 加载谁
- 权重在哪
- 是否用 LoRA/MoE
- 生成参数是多少

##### 第二步：初始化tokenizer和model
- tokenizer 从 `load_from` 来
- model 要么是 MiniMind，要么是 HF 模型
- 再加载 `.pth` 或 HF 权重
- 可选叠加 LoRA

##### 第三步：选择测试模式
- 自动跑内置 prompts
- 或者手工输入聊天

##### 第四步：每轮构造输入
- 更新历史
- 如果是 pretrain 权重：`bos + prompt`
- 否则：`apply_chat_template(conversation, ...)`

##### 第五步：tokenize
把字符串变成：
- `input_ids`
- `attention_mask`

##### 第六步：generate
调用模型自回归生成，同时 streamer 实时打印输出

##### 第七步：后处理
- 截取新生成部分
- decode 成文本
- 保存 assistant 回复到历史
- 统计 tokens/s



#### init_model(args)


#### main()
```
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
```

创建一个命令行参数解析器
`parser.add_argument()`定义这个脚本支持哪些命令行参数

```
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
```
定义一组预设测试问题，便于在自动测试模式下，直接把这组prompt按顺序喂给模型

```
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
```


```
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
```
`input_mode ==0` 直接遍历上面预设的prompts列表
`input_mode != 0` 
`iter(callable, sentinel)` 
- 反复调用callable
- 每次得到一个结果
- 直到结果等于sentinel为止



