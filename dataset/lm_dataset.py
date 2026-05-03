#把不同训练阶段的数据都转换成 PyTorch 训练时可直接使用的样本格式
#SFT / DPO 这类阶段通常只让 assistant 的回答部分参与 loss


#定义了五类数据集，对应5种训练/对齐场景
# (1)PretrainDataset. 输入是普通文本 text，目标是做标准 next-token prediction
# (2)SFTDataset. 用于监督微调（Supervised Fine-Tuning）. 输入是多轮对话 conversations，但通常只让 assistant 回复部分参与 loss
# (3)DPODataset. 用于偏好学习 / DPO. 每条样本有一对回答：chosen：偏好的回答,rejected：不偏好的回答. 需要同时构造两条序列，供 DPO 算法比较
# (4)RLAIFDataset. 用于RL 阶段的 prompt 采样
# (5)AgentRLDataset. 用于Agent/RL 工具调用场景

from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, ClassLabel, Value
#关闭tokenizer并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#对话前处理
# (1)对对话做轻量预处理
# (2)对普通聊天数据，做轻量 system prompt 增强；对 tool 数据，不动

#conversations：一条对话，通常是 list，每个元素是一个 message dict
def pre_processing_chat(conversions, add_system_ratio= 0.2):
    #如果这条对话里任意 message 含有 tools 字段，就直接原样返回，不做修改
    if any(conv.get("tools") for conv in conversions):
        return conversions
    
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]

    if conversions[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role':'system','content':random.choice(SYSTEM_PROMPTS)}]+conversions
    return conversions


#如果 prompt 里存在空思考块,并且随机数大于0.2，就删掉它(把空的思考标签替换为空字符串)
def post_processing_chat(prompt_content, empty_think_ratio= 0.2):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

#原始输入：一个 json/jsonl 数据文件，里面每条样本是一个字典，包含一个文本字段 text
#输出：input_ids 和 labels 两个张量，供语言模型训练使用
class PretrainDataset(Dataset):
    def __init__(self,tokenizer,data_path,max_seq_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.samples = load_dataset('json', data_files=data_path, split='train')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        #self.tokenizer(...)调用分词器，把文本转成模型能处理的token 表示。返回的通常不是单纯一个列表，而是一个tokenizer输出对象，里面有input_ids, attention_mask, token_type_ids等
        #add_special_tokens=False表示不添加特殊标记，truncation=True表示截断，max_length=self.max_seq_length-2表示最大长度减去2，因为要加上开始(BOS)和结束(EOS)标记
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, truncation=True, max_length=self.max_seq_length-2).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels



#"把聊天监督微调数据，处理成语言模型可训练格式”的核心类,是一个监督微调(Supervised Fine-Tuning, SFT)数据集类
#把一条“多轮聊天 JSON 数据”变成语言模型训练时需要的 (input_ids, labels)
#目标是让模型学会按照聊天模板进行多轮对话，根据 user / system 的上下文生成 assistant 回复，根据 user / system 的上下文生成 assistant 回复，重点学 assistant 的回答方式，而不是去“预测用户说的话”
#SFT 的常见做法不是“整条对话每个 token 都监督”，system和user输入都不算loss，而是只让assistant的输出部分参与loss
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        #手工定义数据字段结构
        features = Features({
            'conversations':[{
                'role': Value('string'),
                'content': Value('string'),
                'reasoning_content': Value('string'),
                'tools': Value('string'),
                'tool_calls': Value('string')
            }]})
        #用 Hugging Face datasets 加载 JSON/JSONL 数据
        #features=features：按上面定义的字段格式解析
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features= features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens = False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens = False).input_ids

    
    def __len__(self):
        return len(self.samples)
    

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None

        bad_system_phrases = [
            "minimind",
            "jingyaogong",
            "高效小参数AI模型",
        ]

        for conv in conversations:
            role = conv.get("role", "")
            content = conv.get("content", "")

            # 过滤明显带身份模板的 system
            if role == "system":
                if any(p.lower() in content.lower() for p in bad_system_phrases):
                    continue

            clean_conv = {
                "role": role,
                "content": content
            }

            # 只有确实需要工具调用时才保留
            if conv.get("tool_calls"):
                clean_conv["tool_calls"] = (
                    json.loads(conv["tool_calls"])
                    if isinstance(conv["tool_calls"], str)
                    else conv["tool_calls"]
                )

            messages.append(clean_conv)

        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False
        )
    #只给 assistant 回复打监督标签
    def generate_labels(self, input_ids):
        input_ids_len = len(input_ids)
        labels = [-100]*input_ids_len
        i = 0
        while i < input_ids_len:
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i+len(self.bos_id)
                end = start
                while end < input_ids_len:
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    else:
                        end +=1
                for j in range(start, min(end+len(self.eos_id), input_ids_len)):
                    labels[j] = input_ids[j]
                i = min(end+len(self.eos_id), input_ids_len)
            else:
                i +=1
        return labels
    #去一条聊天数据处理成语言模型训练时需要的 (input_ids, labels)
    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample['conversations']
        # 先处理结构化对话数据
        # 再把结构化对话渲染成字符串 prompt
        # 最后对生成出来的字符串做文本级清洗

        #结构层处理。对话前处理,增加一定概率的system prompt
        conversations = pre_processing_chat(conversations,add_system_ratio=0.0)
        #模板渲染层。创建聊天
        prompt = self.create_chat_prompt(conversations)
        #文本层后处理。对话后处理，删除空思考块
        #post_processing_chat()处理的是空思考块，'<think>\n\n</think>\n\n'，这个并不是原始cnversations中直接写好的，而是chat template渲染后才出现的
        prompt = post_processing_chat(prompt)

        import re

        prompt = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", prompt, flags=re.DOTALL | re.IGNORECASE)
        prompt = re.sub(r"<\s*/?\s*think\s*>", "", prompt, flags=re.IGNORECASE)
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)
        prompt = prompt.strip()

        input_ids = self.tokenizer(prompt).input_ids[:self.max_seq_length]
        input_ids += [self.tokenizer.eos_token_id]*(self.max_seq_length-len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# 把“偏好对比数据”处理成 DPO 训练可直接使用的格式
# 这里的“偏好对比数据”指的是一条样本里不是只有一个标准答案，而是有两个候选回答：一个更好，叫chosen，一个更差，叫rejected

# 一条样本 sample
# ├── chosen   -> chat template -> tokenize -> chosen_input_ids -> chosen_loss_mask
# └── rejected -> chat template -> tokenize -> rejected_input_ids -> rejected_loss_mask

# 然后分别做 shift：
# chosen_input_ids   -> x_chosen, y_chosen, mask_chosen
# rejected_input_ids -> x_rejected, y_rejected, mask_rejected
class DPODataset(Dataset):

    # init初始化：
    # (1)保存tokenizer和最大长度
    # (2)定义padding id
    # (3)定义assisant 回答段的边界标记
    # (4)读取偏好数据集
    def __init__(self, file_path, tokenizer, max_seq_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n').input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n').input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')
    

    def __len__(self):
        return len(self.samples)
    
    #取出一条样本
    # 拿到 chosen 和 rejected
    # 分别做 chat template
    # tokenize
    # 构造 loss mask
    # 再切成自回归训练需要的 x / y / mask

    #样本的变化：原始对话 → prompt 字符串 → token id 序列 → 回答区间 mask → 自回归训练用的 x/y/mask
    # (1)原始对话 list[dict]
# chosen = [
#     {"role": "user", "content": "1+1等于几？"},
#     {"role": "assistant", "content": "1+1等于2。"}
# ]
    # (2)变成prompt字符串，这时还是文本，不是token id序列
# <bos>user
# 1+1等于几？
# <eos>
# <bos>assistant
# 1+1等于2。
# <eos>
    # (3)变成完整 input_ids, 即token id序列
#chosen_loss_mask = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, ...]
    # (4)从完整 input_ids 中提取“回答区域”
# chosen_loss_mask = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, ...]
    # (5)切成自回归训练格式
    def __getitem__(self, index):
        #sample = {
        # "chosen": [
        #         {"role": "user", "content": "介绍一下Transformer"},
        #         {"role": "assistant", "content": "Transformer是一种基于注意力机制的模型。"}
        #     ],
        #     "rejected": [
        #         {"role": "user", "content": "介绍一下Transformer"},
        #         {"role": "assistant", "content": "不知道。"}
        #     ]
        # }
        sample = self.samples[index]
        chosen = sample['chosen']
        rejected = sample['rejected']

        #原始对话 → prompt 字符串
        #apply_chat_template 函数详解是HF库中PreTrainedTokenizer 类提供的一个方法，用于架构对话消息转换成期望的单一字符串或者tokenized输入。
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize = False, add_generation_prompt=False)
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize = False, add_generation_prompt=False)
        rejected_prompt = post_processing_chat(rejected_prompt)

        #prompt字符串 → token id序列
        #这里tokenizer做了三件事：1.把字符串切成了token，2.把token序列转换成token id序列，3.把token id序列填充成固定长度(太长就截断，不够长就pad到max_length)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        #chosen_encoding 不是一个单独列表，而是一个“编码结果字典/对象”。常见字段包括：input_ids, attention_mask。

        #token id 序列 → 回答区间 mask
# chosen_prompt   # 字符串
#       ↓ tokenizer
# chosen_encoding # 编码结果对象
#       ↓ ['input_ids']
# chosen_input_ids  # 整数列表
        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        #回答区间 mask → 自回归训练用的 x/y/mask
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        #和y_chosen对齐
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
        return{
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }



        
    
    # 输入是一整条的token序列，输出是同样长度的0/1序列
    def generate_loss_mask(self, input_ids):
        loss_mask = [0]*len(input_ids)
        i = 0
        input_ids_len = len(input_ids) 
        while i < input_ids_len:
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i+len(self.bos_id)
                end = start
                while end < input_ids_len:
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    else:
                        end +=1
                #前面的bos_id标记为0，后面的eos_id标记为1。bos_id 更像是条件前缀 / 上下文提示。eos_id更像是模型输出的一部分
                for j in range(start, min(end+len(self.eos_id), input_ids_len)):
                    loss_mask[j] = 1
                i = min(end+len(self.eos_id), input_ids_len)
            else:
                i+=1
        return loss_mask
    
#从对话数据里提取“上下文 prompt”，让模型在 RL 阶段自己去生成回答
#SFT/DPO:数据已有回答，训练时通常要对已有回答打分或者监督；RLAIF：训练时只给prompt，让当前模型策略在线生成answer
#Reinforcement Learning from AI Feedback
def RLAIFDataset(Dataset):
    #thinking_ratio: 0.5 表示对每条样本，有 50% 的概率在 prompt 构造时打开“thinking 模式”
    def __init__(self, jsonl_path, tokenizer, max_length = 1024, thinking_ratio = 0.5):
        super().__init__()
        self.thinking_ratio = thinking_ratio
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files = jsonl_path, split = 'train')
    

    def __len__(self):
        return len(self.samples)
    


    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.rand() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(conversations, open_thinking = use_thinking, tokenize = False, add_generation_prompt=True)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        #sample 是整条样本字典,通常长这样：
        # {
        #     "conversations": [
        #         {"role": "system", "content": "你是一个助手"},
        #         {"role": "user", "content": "你叫什么名字？"},
        #         {"role": "assistant", "content": "我叫小A"},
        #         {"role": "user", "content": "你叫什么名字？"},
        #         {"role": "assistant", "content": "小A"},

        prompt = self.create_chat_prompt(sample['conversations'])
        return {
            'prompt': prompt,
            'answer': ""
        }

#把带有工具使用信息的对话样本读进来，拆成“历史消息 + 可用工具 + 标准答案(gt)”三部分，供 agent 强化学习或评估使用。
#它返回的不是token张量，而是一个字典，包含三个key：message， tool， gt
class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        #下面是对samples的手工存取，不同于上面的load_dataset。区别在于，手工读取得到的是python的list[dict],而load_dataset得到的是Hugging Face的Dataset对象
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    

    def __len__(self):
        return len(self.samples)
    

#输入是一条完整的对话，输出是messages和tools
    def parse_conversations(self, conversations):
        messages, tools = [], None
        for conv in conversations:
            conv = dict(conv)
            if conv.get('role')=="system" and conv.get('tools'):
                tools = json.loads(conv['tools']) if isinstance(conv['tools'], str) else conv['tools']
            messages.append(conv)
        return messages[:-1], tools
    
    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}
    
if __name__ == '__main__':
    pass









            



            






        






