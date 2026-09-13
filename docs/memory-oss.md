# 本地开源记忆

此分支使用 Mem0 OSS 的 `Memory`。DeepSeek 从用户表达中提取可复用的事实和约定，本地多语言模型生成向量，Qdrant 将条目保存到磁盘。每轮用户消息到来时，在模型回答前检索当前会话的相关条目，拼入参考上下文。

保存时由 Agent 调用 `add_memory`。成功返回后显示“记忆已更新”；无新增条目显示“本次未新增记忆”；异常显示“记忆更新失败”。这里不再调用 Mem0 Platform，也不需要 `MEM0_key`；DeepSeek 仍使用现有 API。

## 启动

此工作区已配置独立 `.venv`。在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe src/api/app.py
```

也可运行根目录的 `start-memory-oss.cmd`。服务使用 8080 端口；如果旧版本正在占用该端口，先在旧终端按 Ctrl+C。新工作区独立保存会话和记忆，数据库与知识包继续读取当前配置。

## 配置与存储

- `config/secrets.env`：现有 `DEEPSEEK_API_KEY`。
- `DATA_AGENT_MEMORY_ENABLED=1`：启用记忆；设为 0 时不注册保存工具、不检索、不注入记忆。
- `DATA_AGENT_MEMORY_DIR`：默认 `runtime/mem0`，包含 Qdrant 数据与 SQLite 消息历史；重启后保留。
- `DATA_AGENT_MEMORY_LLM_MODEL`：默认 `deepseek-v4-flash`，提取时使用 temperature=0。
- 向量模型：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，CPU，384 维。可用 `DATA_AGENT_MEMORY_EMBEDDING_MODEL` 指向同一模型的本地目录；`DATA_AGENT_MEMORY_EMBEDDING_LOCAL_ONLY=1` 只读取本地缓存。

当前检索使用向量相似度，top_k=10、threshold=0.1，不启用重排、spaCy 实体提取或 BM25。它们是可选组件，不是本次最小接入的前置条件。托管平台的 Dashboard 不显示这些本地条目。

写入的 `run_id` 使用产品的 `thread_id`；检索同时匹配 user_id、workspace_id 与 thread_id。不同会话不共享记忆。当前本地 Qdrant 目录由一个后端进程持有，进程内初始化和读写串行执行。

## 验证

```powershell
.\.venv\Scripts\python.exe src/memory/test_mem0.py
.\.venv\Scripts\python.exe src/memory/test_mem0.py --read-only
```

第一条使用真实 DeepSeek 提取一条虚拟偶像分析约定，然后检查检索及用户、工作空间、会话隔离。第二条在新进程中读取上次数据，不再提交记忆，用于确认持久化。结果保存在独立的 `runtime/mem0-smoke/last_run.json` 和 `restart_check.json`。

## 在另一台机器安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

配置 DeepSeek Key 和现有数据源，首次使用模型名时下载模型。已有完整缓存时可启用 local-only。向量模型固定为上述 384 维配置。

## 分支与评测

`feat/agent-memory-oss` 从 `main` 的 `02ce729` 创建，与托管记忆分支平行。当前压缩、上下文预算和记忆界面的代码作为工作副本带入此分支，原工作区不变。

这次验证只确认接入链路。完整 A/B 需要重新冻结 OSS 配置并从干净状态运行；不能把托管版那次未完成的运行算作开源版效果分数。
