# 健身建议 Skill

## 项目目标

这个新仓库提供一个最小可维护的健身知识库脚手架，让 AI 能够：

- 依据结构化知识块回答训练、动作、营养和安全问题。
- 在回答中优先引用知识块对应的 `source_book` 与 `topic`。
- 遇到疼痛、伤病、孕期、慢性病或极端减脂等高风险场景时停止给处方并建议转介。
- 使用本地向量检索、BM25 和 RRF 融合完成混合检索。
- 通过 Streamlit 提供名为“柯教”的 Web 聊天界面，并调用 DeepSeek 流式生成回答。

当前仓库实现了本地混合检索，不抓取外部内容，也不上传完整知识文件。Web
界面调用 DeepSeek 时，只发送用户问题与本次检索出的 Top 3 知识块。

## 目录结构

```text
docs/       范围、安全边界与知识治理规范
data/       来源登记与知识块模板
knowledge/  知识内容容器
  examples.jsonl  仅用于管线测试的 sample 夹具
  chunks.jsonl    正式知识块容器
tests/      测试问题
scripts/    校验与混合检索脚本
SKILL.md    AI 使用说明
app.py      柯教 Web 聊天界面
```

知识治理的入口是 `docs/KNOWLEDGE_GOVERNANCE.md`。来源信息只在
`data/books.yaml` 维护，知识正文只在 `knowledge/*.jsonl` 维护，不再建立
第二份人工索引。

## 部署位置

本仓库是技能源目录，仓库名可以与技能名不同。Codex 要从以下位置发现并调用它：

- 项目级：项目的 `.agents/skills/fitness-coach/`
- 个人级：`~/.agents/skills/fitness-coach/`

部署时把当前仓库复制或符号链接到上述任一位置，并确保目标目录本身就是技能目录，即 `SKILL.md` 与 `agents/`、`data/`、`docs/`、`knowledge/`、`scripts/`、`tests/` 直接位于该目录下。

在 Windows 上创建符号链接可能需要开发者模式或管理员权限；如果不希望处理权限，直接复制目录即可。当前仓库可作为源目录，后续修改同步到部署位置后再供 Codex 调用。

## 当前假设

1. 运行时路径以项目根目录动态定位，不依赖仓库所在磁盘或绝对路径。
2. 目标用户先假设为健康成年健身新手到进阶训练者。
3. 当前使用本地混合检索，不连接外部抓取服务；Web 生成回答时只向 DeepSeek 发送用户问题与 Top 3 知识块。
4. 示例知识块是通用原则的原创摘要，不声称来自某本书的原文。
5. `knowledge/examples.jsonl` 中当前 10 条记录全部是 `sample: true`、`needs_review: true`，仅用于管线测试。
6. 书单是待建设的来源清单，不代表已经逐本核对或完成版权审阅。
7. 《健身营养全书》和《豹式健身》的责任作者暂时标为“待核验”，需根据正式出版信息补充。
8. 回答使用中文；结构化字段名保留英文。
9. 本技能不处理整本 PDF、不转录书籍、不抓取盗版内容。
10. 本技能不提供医疗诊断、药物或补剂处方。
11. `sample` 是测试夹具的终态，不可升级为 `reviewed`；正式知识块采用 `draft → reviewed → deprecated` 生命周期。
12. 当前正式容器 `knowledge/chunks.jsonl` 包含 58 条已审核知识块，覆盖深蹲、卧推和推举。

## 环境准备

在仓库根目录创建并安装本地虚拟环境。Windows PowerShell 中不要先激活环境，直接调用相对路径的 Python：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果系统没有把 `python` 加入 PATH，可使用 Codex 内置 Python 创建环境：

```powershell
C:\Users\33441\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv .venv
```

依赖文件为：

```text
chromadb
sentence-transformers
rank_bm25
ebooklib
beautifulsoup4
streamlit
openai
```

## 柯教 Web 聊天界面

在仓库根目录启动页面：

```powershell
.\.venv\Scripts\streamlit run app.py
```

启动后在浏览器打开 Streamlit 给出的本地地址，并在侧边栏填写 DeepSeek API
Key。API Key 只保存在当前浏览器会话，不写入仓库。

页面工作流程：

1. 复用 `scripts/retrieve.py` 的查询扩展、向量检索、BM25 和 RRF 融合。
2. Top 1 的 `rrf_score` 低于 `0.02` 时直接提示知识库未覆盖，不调用大模型。
3. 达到阈值后将 Top 3 知识块交给 DeepSeek 生成流式回答。
4. 在回答下方折叠展示 Top 3 的 ID、主题、摘要和来源位置。

仓库已包含 `data/chroma_db/` 和 `data/bm25_index.pkl`，Streamlit Community Cloud
部署时无需重新嵌入知识块。知识内容变化后，再运行 `scripts/build_index.py` 更新并提交索引。

## 混合检索架构

1. `scripts/build_index.py` 读取 `knowledge/chunks.jsonl` 中 `review_status: reviewed` 且 `needs_review: false` 的知识块。
2. 使用 `BAAI/bge-small-zh-v1.5` 将 `content_summary` 转为向量，持久化到 `data/chroma_db/`。
3. 使用同一批知识块构建 BM25 索引，pickle 保存到 `data/bm25_index.pkl`。
4. `scripts/retrieve.py` 同时查询向量 Top 5 和 BM25 Top 5。
5. 使用 Reciprocal Rank Fusion（RRF，`k=60`）融合两路排名，输出最终 Top 3。

知识块变化后必须重新构建索引：

```powershell
.\.venv\Scripts\python.exe scripts/build_index.py
```

混合检索：

```powershell
.\.venv\Scripts\python.exe scripts/retrieve.py "我推举的时候腰酸" --top 3
```

首次构建索引会下载向量模型。如果 Hugging Face 直连失败，可使用镜像后重新构建：

```powershell
$env:HF_ENDPOINT='https://hf-mirror.com'
$env:HF_HUB_DISABLE_XET='1'
.\.venv\Scripts\python.exe scripts/build_index.py
```

检索端默认只读取本地模型缓存，不会在每次查询时重新下载模型。

## 如何添加新知识块

1. 先阅读 `docs/KNOWLEDGE_GOVERNANCE.md`，再从 `data/chunk_template.yaml` 复制字段结构。
2. 确认 `source_book` 对应 `data/books.yaml` 中已有的书籍 `id`。
3. 使用自己的语言写 `content_summary`，不要复制原文。
4. 不确定章节或页码时，把 `source_locator` 写成 `null`。
5. 正式内容先设为 `review_status: draft`、`sample: false`、`needs_review: true`，写入 `knowledge/chunks.jsonl`。
6. 管线测试内容设为 `review_status: sample`、`sample: true`、`needs_review: true`，写入 `knowledge/examples.jsonl`；示例块不得直接升级为正式块。
7. 审核通过后填写 `reviewed_by`、`last_reviewed_at`，把状态改为 `reviewed`。
8. 需要替代已审核内容时，建立新 ID，用 `supersedes` 记录旧 ID，并把旧块改为 `deprecated`。
9. 运行校验脚本。

## 如何运行校验

使用 Python 标准库即可：

```powershell
.\.venv\Scripts\python.exe scripts\validate_chunks.py --expected-examples 10 --expected-chunks 58
```

在 Codex 桌面环境中，也可以使用内置 Python：

```powershell
C:\Users\33441\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts/validate_chunks.py --expected-examples 10 --expected-chunks 58
```

脚本会检查：

- 每行 JSON 是否可解析。
- 必填字段是否存在。
- 枚举字段是否合法。
- `source_book` 是否存在于书单。
- `id` 是否重复。
- `sample: true` 的知识块是否同时设置 `needs_review: true`。
- `sample`、`draft`、`reviewed`、`deprecated` 状态是否与审核标记一致。
- `examples.jsonl` 是否只包含 `sample`，`chunks.jsonl` 是否只包含正式状态。
- `reviewed` 知识块是否包含审核人和 ISO 格式审核时间。
- 废弃知识块是否包含废弃原因。
- 知识块数量是否符合预期。

## 版权与安全注意事项

- 只允许使用原创摘要、结构化原则和必要的最小引用。
- 不嵌入书籍原文，不复制长篇段落，不伪造书籍内容、作者或页码。
- 示例块在人工审核前不能被视为已验证专业结论。
- 疼痛、伤病、孕期、慢性病、进食障碍或极端减脂场景必须转介合格专业人士。
- 本知识库不能替代医疗诊断、康复治疗或注册营养师服务。
