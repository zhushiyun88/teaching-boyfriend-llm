# 👩‍🏫 Teaching Boyfriend LLM (教男朋友学大模型)

<p align="center">
  <img src="https://img.shields.io/badge/LLM-Learning-blue" alt="LLM Learning">
  <img src="https://img.shields.io/github/stars/zhushiyun88/teaching-boyfriend-llm?style=social" alt="Stars">
  <img src="https://img.shields.io/badge/Focus-AGI-orange" alt="AGI">
</p>

## 📖 项目简介

**“我们俩的 LLM 学习笔记，一起迈向 AGI！”**

本项目记录了从零开始学习大语言模型（LLM）的全过程，涵盖了从基础理论、模型架构、训练微调，到 RAG、Agent 以及强化学习等前沿技术。内容主要以 PDF 形式沉淀，适合算法工程师、研究员以及对 LLM 感兴趣的开发者系统性学习。

> 💡 **Tip:** 知识星球有整理好的完整版、大模型求职信息差等，持续更新中！还有对应的时间安排计划表、严选 B 站优质视频和知乎帖，邀请了大厂面试官作为嘉宾。欢迎扫码加入（见底部）！

---

## 🗺️ 学习导航 (Table of Contents)

- [💡 LLM 基础理论](#-llm-基础理论)
- [🏗️ 模型架构与演进](#-模型架构与演进)
- [🔧 训练与微调 (Pre-training & Fine-tuning)](#-训练与微调-pre-training--fine-tuning)
- [🧠 强化学习与对齐 (RL & Alignment)](#-强化学习与对齐-rl--alignment)
- [🔍 RAG 与向量检索](#-rag-与向量检索)
- [🤖 Agent 智能体](#-agent-智能体)
- [📏 长上下文与提示工程](#-长上下文与提示工程)
- [🛠️ 框架与工具 (LangChain/vLLM)](#-框架与工具-langchainvllm)
- [📚 经典算法与杂项](#-经典算法与杂项)

---

### 💡 LLM 基础理论
> 万丈高楼平地起，这里是理解大模型的基石。

- [0514 LLM训练流程与tokenizer](./0514LLM训练流程与tokenizer.pdf)
- [0515 Self Attention与KV Cache](./0515self%20Attention与kv%20cache.pdf)
- [0516 位置编码](./0516位置编码.pdf)
- [0518 Normalize与Decoding的方法](./0518Normalize与decoding的方法.pdf)
- [2025了，如何回答“为什么现在的大模型都是decoder-only的架构”](./25-0502%202025了，如何回答“为什么现在的大模型都是decoder-only的架构”.pdf)

### 🏗️ 模型架构与演进
> 紧跟前沿，深度解析主流开源与闭源模型架构。

**DeepSeek 系列**
- [DeepSeek v3 技术报告精读](./25-0216DeepSeek%20v3技术报告精读.pdf)
- [DeepSeek R1 技术报告解读 (算法工程师视角)](./25-0203算法工程师视角的DeepSeek%20R1技术报告解读.pdf)
- [DeepSeek R1 20 问](./25-0220DeepSeek%20R1%2020%20问.pdf)

**Llama 系列**
- [Llama 3 解析](./0520llama3.pdf)
- [Llama 3.1 技术报告](./0829llama3.1技术报告.pdf)
- [Llama 3 技术报告-后训练](./0918llama3技术报告-后训练.pdf)

**其他前沿模型**
- [Qwen 2.5 系列 (算法工程师视角)](./25-0304算法工程师视角看Qwen2.5系列.pdf)
- [GPT-o3 (算法工程师视角)](./25-0418算法工程师视角看GPT-o3.pdf)
- [Kimi K2](./Kimi%20K2.pdf)
- [MOE (混合专家模型)](./0630MOE.pdf)

### 🔧 训练与微调 (Pre-training & Fine-tuning)
> 从分布式训练到底层优化，再到高效微调技术。

**微调 (PEFT/LoRA)**
- [PEFT (Parameter-Efficient Fine-Tuning)](./0522PEFT（Parameter-Efficient%20Fine-Tuning%20）.pdf)
- [LoRA](./0613%20lora.pdf) | [QLoRA](./0623Qlora.pdf) | [AdaLoRA](./0618%20Adalora.pdf)
- [指令微调](./0601指令微调.pdf) | [指令微调数据集](./0605指令微调数据集.pdf)

**分布式与优化**
- [学习率 (Learning Rate)](./0607学习率（learning%20rate）.pdf)
- [分布式训练1——数据并行](./0728分布式训练1——数据并行.pdf)
- [分布式训练2——DDP](./0730分布式训练2——DDP.pdf)
- [Accelerate](./0803Accelerate.pdf) | [DeepSpeed](./0808deepspeed.pdf)
- [Flash Attention 原理](./0709Flash%20attention-原理.pdf) | [Flash Attention 代码](./0710Flash%20attention-代码.pdf)
- [量化 (Quantization)](./0622Quantization.pdf) | [PTQ](./0704PTQ.pdf)

### 🧠 强化学习与对齐 (RL & Alignment)
> 让模型更符合人类价值观，探索 RLHF 及其变体。

**RL 基础**
- [RL1: 马尔可夫决策过程和贝尔曼方程](./0720强化学习1-马尔可夫决策过程和贝尔曼方程.pdf)
- [RL2: 策略迭代](./0723强化学习2-策略迭代.pdf)
- [RL3: 蒙特卡洛方法](./0806强化学习3蒙特卡洛方法.pdf)

**RLHF & 变体**
- [RLHF](./25-0321RLHF.pdf)
- [PPO](./0819PPO.pdf) | [PPO 演化史](./25-0316PPO是怎么从Policy%20gradient演化而来的.pdf)
- [DPO](./0816DPO.pdf) | [2025 DPO更新](./25-0401DPO.pdf)
- [GRPO](./25-0401GRPO.pdf) | [NSA](./25-0222NSA.pdf) | [DAPO](./25-0401DAPO.pdf)

### 🔍 RAG 与向量检索
> 检索增强生成，解决大模型幻觉与知识时效性问题。

**RAG 基础与实战**
- [RAG 入门](./0524RAG入门.pdf)
- [RAG from Scratch (LangChain)](./0526RAG%20for%20Scratch——langchain.pdf)
- [GraphRAG](./0715GraphRAG.pdf) | [2025 GraphRAG](./25-0302GraphRAG.pdf)
- [Agentic RAG](./25-0421Agentic%20RAG.pdf) | [Agentic RAG 案例分析](./25-0421Agentic%20RAG案例分析.pdf)

**Embedding & 检索**
- [Embedding Model 1](./1124embedding%20model1.pdf) | [Embedding Model 2](./1206embedding%20model2.pdf)
- [向量索引](./1203向量索引.pdf)
- [Rerank (重排序)](./1212Rerank1.pdf)

### 🤖 Agent 智能体
> 通向 AGI 的重要路径：规划、记忆与工具调用。

- [Agent 概述](./25-0307Agent概述.pdf) | [Agent 入门](./1117Agent入门.pdf)
- [Function Call](./25-0507Function%20call.pdf)
- [MCP (Model Context Protocol)](./25-0501MCP.pdf)
- [Deep Research](./25-0228Deep%20research.pdf)
- [那些年失败的多智能体](./25-0502那些年失败的多智能体.pdf)
- [实战：阿里云百炼开发智能导购 Agent](./25-0326如何用阿里云百炼开发智能导购Agent？.pdf)

**Planning & Memory**
- [Planning 1: 基础方法](./1220Agent%20Planning1%20基础方法.pdf)
- [Planning 2: 规划](./1223Agent%20Planning2%20规划.pdf)
- [Planning 3: 反思](./25-0107Agent%20Planning3%20反思.pdf)
- [Agent Memory](./1230Agent%20memory.pdf)
- [Memory-based Agent](./25-0121Memory-based%20agent.pdf) | [Part 2](./25-0127Memory-based%20agent2.pdf)

### 📏 长上下文与提示工程
> 突破 Token 限制与更好地与模型对话。

- [如何写出优雅的 Prompt](./0827如何写出优雅的Prompt.pdf)
- [Long Context: 插值](./1010Long%20context2-%20插值(1).pdf)
- [Long Context: 上下文窗口分割](./1016Long%20context3-上下文窗口分割(1)(1).pdf)
- [Long Context: 提示压缩](./1022Long%20context4%20提示压缩1.pdf)

### 🛠️ 框架与工具 (LangChain/vLLM)
- [vLLM 入门](./0813vllm入门.pdf)
- [LangChain 介绍与模型组件](./1104LangChain%20介绍与模型组件.pdf)
- [LangChain 提示工程](./1110LangChain2%20提示工程.pdf)
- [LangChain 模型调用和输出解析](./1111LangChain3%20模型调用和输出解析.pdf)

### 📚 经典算法与杂项
- [XGBoost](./0718XGBoost.pdf)
- [MOCO](./0611MOCO.pdf) | [重读经典 MOCO](./重读经典：MOCO.pdf)

---

## 🤝 交流与加入

如果你觉得这些笔记对你有帮助，欢迎 **Star** ⭐️ 支持！

**加入我们：**
我们俩的llm学习笔记，一起迈向AGI！

知识星球有整理好的完整版，大模型求职信息差等，持续更新中！ 还有对应的时间安排计划表、严选b站优质视频和知乎帖，邀请了大厂面试官作为嘉宾。 欢迎扫码加入！

![微信图片_20250508120115](https://github.com/user-attachments/assets/b95ff628-8eea-4715-97a6-8baa7e04e4a5)
