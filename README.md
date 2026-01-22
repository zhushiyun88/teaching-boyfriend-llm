



<div align="center">

# 🧠 Teaching Boyfriend LLM

<p>
  <a href="https://github.com/zhushiyun88/teaching-boyfriend-llm"><img alt="GitHub stars" src="https://img.shields.io/github/stars/zhushiyun88/teaching-boyfriend-llm?style=flat-square&logo=github"></a>
  <a href="https://github.com/zhushiyun88/teaching-boyfriend-llm/fork"><img alt="GitHub forks" src="https://img.shields.io/github/forks/zhushiyun88/teaching-boyfriend-llm?style=flat-square&logo=github"></a>
  <a href="https://github.com/zhushiyun88/teaching-boyfriend-llm/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/zhushiyun88/teaching-boyfriend-llm?style=flat-square&logo=github"></a>
</p>

**「教男朋友学LLM」** 一份系统性的大语言模型学习指南

从LLM基础原理到前沿技术，涵盖预训练、微调、RAG、Agent、推理优化等核心知识

</div>

---

## 📖 目录

- [项目介绍](#-项目介绍)
- [学习路线](#-学习路线)
- [核心内容](#-核心内容)
  - [一、LLM 基础原理](#一llm-基础原理)
  - [二、模型微调 Fine-tuning](#二模型微调-fine-tuning)
  - [三、强化学习与对齐 RLHF](#三强化学习与对齐-rlhf)
  - [四、RAG 检索增强生成](#四rag-检索增强生成)
  - [五、Agent 智能体](#五agent-智能体)
  - [六、LangChain 框架](#六langchain-框架)
  - [七、分布式训练](#七分布式训练)
  - [八、推理优化](#八推理优化)
  - [九、Long Context 长上下文](#九long-context-长上下文)
  - [十、Embedding & 向量检索](#十embedding--向量检索)
  - [十一、前沿模型技术报告](#十一前沿模型技术报告)
  - [十二、提示工程 Prompt Engineering](#十二提示工程-prompt-engineering)
  - [十三、其他高级主题](#十三其他高级主题)
- [贡献指南](#-贡献指南)
- [Star 趋势](#-star-趋势)

---

## 🌟 项目介绍

这是一份**系统性**的大语言模型 (LLM) 学习资料库，旨在帮助初学者从零开始理解 LLM 的核心原理与前沿技术。

### ✨ 项目特点

- 📚 **系统全面** - 覆盖从基础到进阶的完整知识体系
- 🎯 **循序渐进** - 按日期顺序编排，学习路径清晰
- 💡 **深入浅出** - 复杂概念用通俗易懂的方式讲解
- 🔥 **紧跟前沿** - 包含 DeepSeek、Qwen、GPT-o3 等最新技术解读
- 🛠️ **理论+实践** - 原理讲解与代码实现相结合

### 👥 适用人群

- 🎓 想要入门 LLM 领域的开发者
- 💼 准备转型 AI/LLM 方向的工程师
- 📖 希望系统学习大模型知识的学生
- 🔬 需要快速了解前沿技术的研究者

---

## 📚 核心内容

### 一、LLM 基础原理

> 🔖 **难度：⭐⭐ | 推荐优先级：必学**

| 文档 | 核心内容 |
|------|---------|
| [0514 LLM训练流程与tokenizer](0514LLM训练流程与tokenizer.pdf) | LLM 训练的完整流程、Tokenizer 的原理与实现 |
| [0515 Self Attention与KV Cache](0515self%20Attention与kv%20cache.pdf) | 自注意力机制原理、KV Cache 加速推理 |
| [0516 位置编码](0516位置编码.pdf) | 绝对位置编码、相对位置编码、RoPE |
| [1013 位置编码](1013位置编码.pdf) | 位置编码进阶讲解 |
| [0518 Normalize与Decoding方法](0518Normalize与decoding的方法.pdf) | LayerNorm、RMSNorm、各种解码策略 |
| [0520 LLaMA3](0520llama3.pdf) | LLaMA3 模型架构详解 |
| [0607 学习率](0607学习率（learning%20rate）.pdf) | 学习率调度策略、Warmup、Cosine Decay |
| [预训练](预训练.pdf) | 大模型预训练完整流程 |
| [为什么大模型都是 Decoder-only 架构](25-0502%202025了，如何回答"为什么现在的大模型都是decoder-only的架构".pdf) | Decoder-only 架构优势分析 |

### 二、模型微调 Fine-tuning

> 🔖 **难度：⭐⭐⭐ | 推荐优先级：必学**

| 文档 | 核心内容 |
|------|---------|
| [0522 PEFT 参数高效微调](0522PEFT（Parameter-Efficient%20Fine-Tuning%20）.pdf) | PEFT 概述、各种高效微调方法对比 |
| [0601 指令微调](0601指令微调.pdf) | Instruction Tuning 原理与实践 |
| [0605 指令微调数据集](0605指令微调数据集.pdf) | 高质量指令数据集构建方法 |
| [0613 LoRA](0613%20lora.pdf) | Low-Rank Adaptation 原理与实现 |
| [0618 AdaLoRA](0618%20Adalora.pdf) | 自适应 LoRA 参数分配 |
| [0622 Quantization](0622Quantization.pdf) | 模型量化技术：INT8/INT4 量化 |
| [0623 QLoRA](0623Qlora.pdf) | 量化 + LoRA 联合优化 |
| [0704 PTQ](0704PTQ.pdf) | Post-Training Quantization 训练后量化 |

### 三、强化学习与对齐 RLHF

> 🔖 **难度：⭐⭐⭐⭐ | 推荐优先级：进阶必学**

#### 3.1 强化学习基础

| 文档 | 核心内容 |
|------|---------|
| [0720 强化学习1 - MDP与贝尔曼方程](0720强化学习1-马尔可夫决策过程和贝尔曼方程.pdf) | 马尔可夫决策过程、贝尔曼方程 |
| [0723 强化学习2 - 策略迭代](0723强化学习2-策略迭代.pdf) | 策略迭代、值迭代算法 |
| [0806 强化学习3 - 蒙特卡洛方法](0806强化学习3蒙特卡洛方法.pdf) | MC 方法、TD 方法 |

#### 3.2 对齐算法

| 文档 | 核心内容 |
|------|---------|
| [0816 DPO](0816DPO.pdf) | Direct Preference Optimization |
| [0819 PPO](0819PPO.pdf) | Proximal Policy Optimization |
| [25-0316 PPO 演化历程](25-0316PPO是怎么从Policy%20gradient演化而来的.pdf) | 从 Policy Gradient 到 PPO |
| [25-0321 RLHF](25-0321RLHF.pdf) | RLHF 完整流程详解 |
| [25-0401 DPO](25-0401DPO.pdf) | DPO 进阶讲解 |
| [25-0401 GRPO](25-0401GRPO.pdf) | Group Relative Policy Optimization |
| [25-0401 DAPO](25-0401DAPO.pdf) | Diffusion-based Alignment |
| [GFPO](GFPO.pdf) | Guided Flow Policy Optimization |
| [GSPO](GSPO：从token级到序列级优化的范式转变.pdf) | 从 Token 级到序列级优化 |
| [SAPO](SAPO.pdf) | Self-Alignment Policy Optimization |
| [大模型强化学习中的熵机制](大模型在强化学习中探索和学习的熵机制.pdf) | 熵正则化在 RLHF 中的作用 |

### 四、RAG 检索增强生成

> 🔖 **难度：⭐⭐⭐ | 推荐优先级：应用必学**

| 文档 | 核心内容 |
|------|---------|
| [0524 RAG 入门](0524RAG入门.pdf) | RAG 基础概念与架构 |
| [0526 RAG from Scratch - LangChain (1)](0526RAG%20for%20Scratch——langchain.pdf) | 用 LangChain 从零实现 RAG |
| [0528 RAG from Scratch - LangChain (2)](0528Rag%20for%20Scratch——langchain.pdf) | RAG 进阶实现 |
| [0530 RAG from Scratch - LangChain (3)](0530Rag%20for%20Scratch——langchain.pdf) | RAG 高级技巧 |
| [0715 GraphRAG](0715GraphRAG.pdf) | 图结构增强的 RAG |
| [25-0302 GraphRAG](25-0302GraphRAG.pdf) | GraphRAG 深入讲解 |
| [25-0421 Agentic RAG](25-0421Agentic%20RAG.pdf) | Agent + RAG 融合架构 |
| [25-0421 Agentic RAG 案例分析](25-0421Agentic%20RAG案例分析.pdf) | Agentic RAG 实战案例 |

### 五、Agent 智能体

> 🔖 **难度：⭐⭐⭐⭐ | 推荐优先级：前沿方向**

#### 5.1 Agent 基础

| 文档 | 核心内容 |
|------|---------|
| [1117 Agent 入门](1117Agent入门.pdf) | Agent 基础概念与架构 |
| [25-0307 Agent 概述](25-0307Agent概述.pdf) | Agent 技术全景图 |
| [25-0507 Function Call](25-0507Function%20call.pdf) | 函数调用机制 |
| [25-0501 MCP](25-0501MCP.pdf) | Model Context Protocol |

#### 5.2 Agent Planning

| 文档 | 核心内容 |
|------|---------|
| [1220 Agent Planning1 - 基础方法](1220Agent%20Planning1%20基础方法.pdf) | 规划基础方法 |
| [1223 Agent Planning2 - 规划](1223Agent%20Planning2%20规划.pdf) | 高级规划策略 |
| [25-0107 Agent Planning3 - 反思](25-0107Agent%20Planning3%20反思.pdf) | Reflection 机制 |

#### 5.3 Agent Memory

| 文档 | 核心内容 |
|------|---------|
| [1230 Agent Memory](1230Agent%20memory.pdf) | Agent 记忆机制 |
| [25-0121 Memory-based Agent (1)](25-0121Memory-based%20agent.pdf) | 记忆驱动的 Agent |
| [25-0127 Memory-based Agent (2)](25-0127Memory-based%20agent2.pdf) | Memory Agent 进阶 |
| [Engram](Engram.pdf) | Engram 记忆架构 |

#### 5.4 Agent 实战

| 文档 | 核心内容 |
|------|---------|
| [25-0326 阿里云百炼智能导购 Agent](25-0326如何用阿里云百炼开发智能导购Agent？.pdf) | Agent 开发实战 |
| [25-0502 失败的多智能体](25-0502那些年失败的多智能体.pdf) | 多智能体系统经验教训 |

### 六、LangChain 框架

> 🔖 **难度：⭐⭐ | 推荐优先级：应用必学**

| 文档 | 核心内容 |
|------|---------|
| [1104 LangChain 介绍与模型组件](1104LangChain%20介绍与模型组件.pdf) | LangChain 基础与架构 |
| [1110 LangChain2 - 提示工程](1110LangChain2%20提示工程.pdf) | LangChain 中的 Prompt 管理 |
| [1111 LangChain3 - 模型调用与输出解析](1111LangChain3%20模型调用和输出解析.pdf) | LLM 调用与输出解析器 |

### 七、分布式训练

> 🔖 **难度：⭐⭐⭐⭐ | 推荐优先级：工程必学**

| 文档 | 核心内容 |
|------|---------|
| [0728 分布式训练1 - 数据并行](0728分布式训练1——数据并行.pdf) | DP、DDP 原理 |
| [0730 分布式训练2 - DDP](0730分布式训练2——DDP.pdf) | PyTorch DDP 实现细节 |
| [0803 Accelerate](0803Accelerate.pdf) | HuggingFace Accelerate 使用 |
| [0808 DeepSpeed](0808deepspeed.pdf) | DeepSpeed ZeRO 优化 |

### 八、推理优化

> 🔖 **难度：⭐⭐⭐⭐ | 推荐优先级：工程必学**

#### 8.1 Attention 优化

| 文档 | 核心内容 |
|------|---------|
| [0709 Flash Attention - 原理](0709Flash%20attention-原理.pdf) | Flash Attention 原理详解 |
| [0710 Flash Attention - 代码](0710Flash%20attention-代码.pdf) | Flash Attention 代码实现 |
| [PageAttention](Pageattention.pdf) | vLLM PagedAttention 原理 |

#### 8.2 推理服务

| 文档 | 核心内容 |
|------|---------|
| [0813 vLLM 入门](0813vllm入门.pdf) | vLLM 高性能推理框架 |
| [Continuous Batching](Continuous%20batching.pdf) | 连续批处理技术 |
| [Prefill 与 Decode](Prefill与Decode.pdf) | 预填充与解码分离 |
| [DistServe 预填充解码解耦](DistServe：预填充和解码解耦.pdf) | 分布式推理优化 |
| [SARATHI Chunked Prefill](SARATHI%20之Chunked%20Prefill.pdf) | 分块预填充技术 |

#### 8.3 推测解码

| 文档 | 核心内容 |
|------|---------|
| [投机解码 Speculative Decoding](投机解码Speculative%20Decoding.pdf) | 推测解码加速推理 |
| [Medusa](Medusa.pdf) | 多头推测解码 |
| [为什么推理阶段是左 Padding](为什么现在大模型在推理阶段都是左padding？.pdf) | Left Padding 原理 |

### 九、Long Context 长上下文

> 🔖 **难度：⭐⭐⭐⭐ | 推荐优先级：进阶**

| 文档 | 核心内容 |
|------|---------|
| [1010 Long Context2 - 插值](1010Long%20context2-%20插值(1).pdf) | 位置编码插值扩展 |
| [1016 Long Context3 - 上下文窗口分割](1016Long%20context3-上下文窗口分割%20(1)(1).pdf) | 长文本分块处理 |
| [1022 Long Context4 - 提示压缩 (1)](1022Long%20context4%20提示压缩1.pdf) | Prompt Compression |
| [1025 Long Context4 - 提示压缩 (2)](1025Long%20context4%20提示压缩2.pdf) | 高级压缩技术 |

### 十、Embedding & 向量检索

> 🔖 **难度：⭐⭐⭐ | 推荐优先级：应用必学**

| 文档 | 核心内容 |
|------|---------|
| [1124 Embedding Model (1)](1124embedding%20model1.pdf) | Embedding 模型原理 |
| [1206 Embedding Model (2)](1206embedding%20model2.pdf) | Embedding 模型进阶 |
| [1203 向量索引](1203向量索引.pdf) | FAISS、向量数据库 |
| [1212 Rerank](1212Rerank1.pdf) | 重排序模型 |

### 十一、前沿模型技术报告

> 🔖 **难度：⭐⭐⭐⭐ | 推荐优先级：保持前沿**

#### 11.1 LLaMA 系列

| 文档 | 核心内容 |
|------|---------|
| [0829 LLaMA 3.1 技术报告](0829llama3.1技术报告.pdf) | LLaMA 3.1 技术详解 |
| [0918 LLaMA 3 后训练](0918llama3技术报告-后训练.pdf) | LLaMA 3 后训练技术 |

#### 11.2 DeepSeek 系列

| 文档 | 核心内容 |
|------|---------|
| [25-0203 DeepSeek R1 技术报告](25-0203算法工程师视角的DeepSeek%20R1技术报告解读.pdf) | DeepSeek R1 深度解读 |
| [25-0216 DeepSeek V3 技术报告](25-0216DeepSeek%20v3技术报告精读.pdf) | DeepSeek V3 精读 |
| [25-0220 DeepSeek R1 20问](25-0220DeepSeek%20R1%2020%20问.pdf) | R1 技术问答 |
| [重构残差连接: DeepSeek mHC](重构残差连接：DeepSeek%20mHC%20架构中的几何与数学原理.pdf) | mHC 架构深度解析 |

#### 11.3 Qwen 系列

| 文档 | 核心内容 |
|------|---------|
| [25-0304 Qwen2.5 系列](25-0304算法工程师视角看Qwen2.5系列.pdf) | Qwen2.5 技术解读 |
| [Qwen3-VL 技术报告](Qwen3-VL技术报告解读.pdf) | Qwen3 视觉语言模型 |
| [Qwen3-VL 核心技术](Qwen3VL核心技术解析.pdf) | Qwen3-VL 核心解析 |

#### 11.4 其他前沿模型

| 文档 | 核心内容 |
|------|---------|
| [25-0418 GPT-o3](25-0418算法工程师视角看GPT-o3.pdf) | GPT-o3 技术分析 |
| [Kimi K2](Kimi%20K2.pdf) | Kimi K2 模型解读 |

### 十二、提示工程 Prompt Engineering

> 🔖 **难度：⭐⭐ | 推荐优先级：应用必学**

| 文档 | 核心内容 |
|------|---------|
| [0827 如何写出优雅的 Prompt](0827如何写出优雅的Prompt.pdf) | Prompt 最佳实践 |
| [Chain of Draft](Chain%20of%20Draft.pdf) | CoD 思维链草稿 |
| [Stop Overthinking](Stop%20Overthinking.pdf) | 避免过度推理 |

### 十三、其他高级主题

> 🔖 **难度：⭐⭐⭐⭐ | 推荐优先级：按需学习**

#### 13.1 MoE 混合专家

| 文档 | 核心内容 |
|------|---------|
| [0630 MoE](0630MOE.pdf) | Mixture of Experts 原理 |

#### 13.2 对比学习

| 文档 | 核心内容 |
|------|---------|
| [0611 MOCO](0611MOCO.pdf) | MOCO 对比学习 |
| [重读经典: MOCO](重读经典：MOCO.pdf) | MOCO 深度解读 |

#### 13.3 Deep Research

| 文档 | 核心内容 |
|------|---------|
| [25-0228 Deep Research](25-0228Deep%20research.pdf) | Deep Research 方法论 |
| [Deep Research](Deep%20research.pdf) | 深度研究技术 |

#### 13.4 其他

| 文档 | 核心内容 |
|------|---------|
| [0718 XGBoost](0718XGBoost.pdf) | XGBoost 算法详解 |
| [25-0222 NSA](25-0222NSA.pdf) | Neural Scaling Analysis |

---

## 🤝 贡献指南

欢迎参与贡献！你可以通过以下方式参与：

1. **🐛 报告问题** - 发现文档错误或有建议？请提交 Issue
2. **📝 完善文档** - 补充内容、修正错误、改进表述
3. **🌟 Star 支持** - 如果本项目对你有帮助，请点个 Star ⭐

### 提交 PR 流程

1. Fork 本仓库
2. 创建你的分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 📊 Star 趋势

如果你觉得这个项目对你有帮助，请给个 Star ⭐️ 支持一下！

[![Star History Chart](https://api.star-history.com/svg?repos=zhushiyun88/teaching-boyfriend-llm&type=Date)](https://star-history.com/#zhushiyun88/teaching-boyfriend-llm&Date)

---

## 📜 License

本项目采用 [MIT License](LICENSE) 开源协议。

---

<div align="center">

**如果觉得有帮助，别忘了给个 ⭐ Star 哦！**

Made with ❤️ by [zhushiyun88](https://github.com/zhushiyun88)

</div>

欢迎加入我的知识星球，星球提供以下资源：

✅ 1. 系统性强，覆盖求职全路径
星球提供了一套完整的大模型求职解决方案，从知识构建、学习计划、面试准备到实战项目，层层递进。尤其适合转行或基础较弱的同学，据说已帮助500+人成功入职算法岗，战绩可查！


✅ 2. 35w字大模型笔记 + 可执行学习计划
笔记内容极其详实（35万字！），且持续更新，支持导出PDF，方便离线学习。更贴心的是，提供了两种学习计划表——“速成版”和“稳扎稳打版”，每天只需4小时，高效无压力。
 

✅ 3. 面试题库口语化、易记忆
收录超300道大厂真题，答案摒弃复杂公式图表，改用口语化表达，更容易理解和背诵。对面试紧张、不擅长技术表达的同学非常友好！
 

✅ 4. 实战项目+简历指导+面试逐字稿
不仅有企业级Agent项目代码，还教你如何写简历、如何讲解项目，甚至提供“面试逐字稿”——这种细节级的辅导在市面上很少见，真正做到了“求职无死角”。
 

✅ 5. 打破信息差，解答真实困惑
涵盖了求职中最常见的痛点：项目速成、微调深入、场景题回答、offer选择等……这些都是过来人最懂的问题，能帮你少走很多弯路。
 

✅ 6. 嘉宾团队强大，可无限提问
导师来自同济、北大、北航、字节等顶尖院校和企业，覆盖大模型、搜广推、AInfra、CV等多个方向，甚至还有万star GitHub项目核心贡献者！更重要的是——可以无限次提问，这性价比简直拉满。
 

如果你正在焦虑如何转型算法岗、如何系统学习大模型，或者只是想找一个能实时答疑的高质量圈子，不妨扫码加入试试，不合适三天内（72h）无理由退款。真诚推荐给每一位努力前行的小伙伴！
![717746009460543cff3903226108e4b0](https://github.com/user-attachments/assets/8ec7a8c6-e123-4088-ac65-9a68e3ccfd28)

