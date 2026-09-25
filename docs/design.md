# traj-analyzer 设计文档

## 1. 目标

traj-analyzer 对一批任意格式的 LLM trajectory 做批量分析。
它按一组可配置的特征把每条 trajectory 转换成数值，在特征空间上采样出少量最值得阅读的 trajectory，再由 Claude Code 阅读这些样本并撰写 insight 报告。

使用者包括人和 agent。
所有命令都以非交互方式执行，stdout 只输出一个 JSON 文档，日志写到 stderr。

## 2. 总体流程

```mermaid
flowchart TD
    raw[原始文件] -->|adapter| unified[统一 Trajectory]
    unified -->|render 与 chunk| md["&lt;id&gt;.md，带步号和分片标记"]
    md -->|traj discover| propose[Claude Code 阅读样本并提议特征]
    propose --> yaml["features/*.yaml，人通过 git diff 审阅"]
    yaml -->|traj extract| table[特征表，每个特征组一个文件]
    md -->|traj extract| table
    table -->|vectorize| frame[宽表与距离矩阵]
    frame -->|traj sample| picks[入选列表，每条带入选原因]
    picks --> report["Claude Code 阅读样本，写入 reports/*.md"]
    report -->|修改定义并提交| yaml
```

工作分为两层：

| 层 | 负责的工作 | 形式 |
|---|---|---|
| `traj` CLI | 读入、渲染、分片、特征提取调度、存储、向量化、采样 | Python 包，执行过程确定 |
| Claude Code skill | 特征发现、阅读样本、撰写报告、把反馈写回 YAML | 分析项目中的 `.claude/skills/` |

## 3. 分析项目

工具仓库和分析项目相互独立。
分析项目是一个 git 仓库，由 `traj init <dir>` 生成，包含以下内容：

| 路径 | 内容 | 是否进入 git |
|---|---|---|
| `traj.yaml` | 数据集、渲染、engine、提取并发 | 是 |
| `features/<group>.yaml` | 特征组定义，每个文件一个组 | 是 |
| `samplers/<name>.yaml` | 采样策略 | 是 |
| `adapters/*.py` | 项目自带的适配器 | 是 |
| `reports/*.md` | Claude Code 撰写的报告 | 是 |
| `.claude/skills/` | `traj-discover` 和 `traj-report` 两个 skill | 是 |
| `.traj/datasets/<dataset>/` | `<id>.json`、`<id>.md`、`index.jsonl` | 否 |
| `.traj/instructions/` | 由特征组生成的 instruction 文件 | 否 |
| `.traj/mailbox/` | aifn 的任务与结论，同时作为缓存 | 否 |
| `.traj/features/<group>.jsonl` | 特征表 | 否 |
| `.traj/samples/<sampler>/<run>/selection.json` | 采样结果 | 否 |

特征定义、采样策略和报告的每次变化都可以通过 diff 和 commit message 追溯。
`.traj/` 中的内容都可以由定义文件和原始数据重新生成。

## 4. 统一 Trajectory

| 类型 | 字段 | 说明 |
|---|---|---|
| Trajectory | `id` | 数据集内唯一 |
| | `dataset` | 数据集名 |
| | `metadata` | 任意键值，例如 cwd、模型、标题、reward |
| | `steps` | Step 列表 |
| Step | `index` | 从 0 开始的步号 |
| | `role` | `user`、`assistant`、`tool`、`system` |
| | `kind` | `message`、`thinking`、`tool_call`、`tool_result` |
| | `content` | 文本 |
| | `name` | 工具名，用于 `tool_call` 和 `tool_result` |
| | `is_error` | `tool_result` 是否报错 |
| | `timestamp` | 时间戳 |

trajectory 的全局标识是 `<dataset>/<id>`，文档中称为 key。

### 4.1 适配器

适配器把原始文件转换成 Trajectory。
`traj.yaml` 的 `datasets` 为每个数据集指定适配器、输入 glob 列表、排除 glob 列表和适配器参数。
适配器可以是内置名称、`module:Class`，或者项目内的 `adapters/foo.py:Class`。

内置的 `claude_code` 适配器读取 Claude Code session 文件。
它按文件顺序读取 `user` 和 `assistant` 记录，所以通过回退放弃的对话轮次也保留在 trajectory 中。
`origin.kind` 不是 `human` 的 user 记录，例如后台任务通知，转换为 `system` 步骤。
以 `<command-name>`、`<local-command-stdout>`、`<bash-input>` 等标签开头的 user 文本是本地命令的记录，同样转换为 `system` 步骤。
它支持的内容块类型是 `text`、`image`、`thinking`、`tool_use`、`tool_result` 和 `fallback`，其他类型会使读入过程报错终止。
为空的 `thinking` 块不产生步骤。

内置的 `messages` 适配器读取每条记录带一个 `{role, content}` 消息列表的数据。
消息列表可以是 JSON 数组，也可以是 JSON 字符串。
`role_map` 把每个源角色映射到统一的 `role` 和 `kind`，出现未映射的角色时读入过程报错终止。
`tool_call_format` 指定 `tool_call` 消息的内容如何编码工具名，取值为 `text`、`json` 或 `python_literal`。

读入时遇到损坏的文件会报错终止，并给出文件路径和行号。
需要跳过的文件写入该数据集的 `exclude` 列表。

### 4.2 渲染与分片

渲染把 Trajectory 写成 Markdown，每步一个标题 `### #12 assistant · tool_call · Bash`。
超过 `render.max_step_chars` 的内容会被截断，并注明截断的字符数。

分片在步与步之间切分，每片不超过 `render.chunk_chars` 个字符，片头写入 `<!-- chunk 3 -->` 标记。
单个步骤超过上限时独占一片。

## 5. 特征

### 5.1 特征组

特征按组定义。
一个组对应一个 aifn `AiFunction`，一次调用输出组内所有特征，所以每条 trajectory 在一个组内只被阅读一次。

```yaml
group: friction
description: 用户与 agent 协作中的摩擦
engine: deepseek
evidence: true
guidance: |
  判断时只看用户的原话，agent 对自身工作的描述不作为依据。
features:
  - name: user_frustration
    type: scalar
    range: [0, 1]
    thresholds: {high: 0.7}
    description: 用户对 agent 表现出的不满程度，0 表示没有不满。
  - name: frustration_curve
    type: vector
    per: chunk
    range: [0, 1]
    description: 每个分片内用户的不满程度。
  - name: task_type
    type: category
    labels:
      bugfix: 修复已有缺陷
      feature: 新增功能
      question: 只提问，不修改代码
      other: 其他
```

`engine` 取值为 `deepseek` 或 `builtin`，默认是 `deepseek`。
`evidence` 为 true 时，每个特征附带一段引用步号的判断依据。
特征名在整个项目内唯一，因为它们会成为宽表的列名。

### 5.2 特征类型

| type | 输出 | 必需字段 | 可选字段 |
|---|---|---|---|
| `scalar` | 浮点数 | | `range` |
| `boolean` | 布尔值 | | |
| `category` | 单个标签 | `labels` | |
| `set` | 互不相同的标签列表 | | `labels`，缺省时为自由字符串 |
| `vector` | 浮点数列表 | `per: chunk` 或 `length: k` | `range` |
| `distribution` | 标签到概率的映射，总和为 1 | `labels` | |

所有类型都可以带 `thresholds`，采样条件按名称引用这些阈值。

### 5.3 从 YAML 生成 aifn 函数

1. `pydantic.create_model` 把特征组转换成 Pydantic 输出模型，`range`、`labels`、元素互不相同和概率总和都成为校验规则。
   agent 提交的结果不合法时，aifn 把字段路径和错误信息反馈给 agent，由 agent 修正后重新提交。
2. 特征定义渲染成 `.traj/instructions/<group>.md`，内容包括任务说明、文件结构、每个特征的定义和取值约束、依据的写法。
3. 请求内容是 `{trajectory_file, n_chunks, sha256}`。
   workspace 是该数据集的渲染目录，权限为只读。
4. aifn 的 mailbox 对函数名、输入、instruction 和 workspace 都相同的请求直接复用已有结论。
   请求中包含渲染文件的内容哈希，instruction 由特征定义生成，所以数据或定义变化时会重新计算，其余部分直接使用缓存。

`engine: builtin` 的组不调用 LLM，每个特征通过 `fn` 调用 `traj_analyzer.features.builtin` 中登记的函数。
内置函数包括 `n_steps`、`n_user_turns`、`n_tool_calls`、`tool_error_rate`、`total_chars`、`duration_minutes` 和 `tools_used`。

### 5.4 提取调度

`traj extract` 按以下步骤执行：

1. 检查 `engine.env_passthrough` 中的环境变量是否都已设置，缺少时报错终止。
   然后为每条 trajectory 和每个 LLM 特征组向 mailbox 提交任务，已有结论的任务直接复用。
2. 存在未完成的任务时，启动 `extract.workers` 个 `python -m aifn worker traj_analyzer.runtime:make_worker --once` 子进程。
   worker 通过环境变量 `TRAJ_PROJECT` 找到项目，并由同一份 YAML 构建相同的函数表。
   队列清空后 worker 退出，任何 worker 以非零状态码退出时命令报错终止。
3. 读取全部结论，写入 `.traj/features/<group>.jsonl`，每行是 `{key, group, feature, value, evidence, status, detail, spec_hash, sha256}`。
   `spec_hash` 是特征组定义的哈希，`sha256` 是计算时渲染文件的内容哈希。
   拒答写为 `refused`，执行失败写为 `failed`，逐片向量长度与分片数不一致写为 `invalid_length`。

`--dry-run` 只查询缓存，不向 mailbox 写入任务。
worker 每次运行都会处理 mailbox 中全部未完成的任务，包括之前中断的运行留下的任务。

### 5.5 模型配置

`traj.yaml` 的 `engine` 段决定 LLM 特征由哪个模型计算：

| 字段 | 作用 |
|---|---|
| `dsh_home` | 安装了 aifn harness bundle 的 dsh home |
| `provider`、`model` | dsh 中的 provider 名和模型名，特征组可以用 `model` 字段覆盖模型名 |
| `max_tokens`、`reasoning_effort` | 传给模型的生成参数 |
| `env_passthrough` | 传给 dsh 运行时的环境变量名，通常是 API key |
| `patches` | 相对项目根目录的 dsh patch 文件列表，用于添加 provider 路由 |

通过 OpenAI 兼容接口访问模型时，在 patch 中配置 `sdk` profile 的 `llm-pi-ai` 行：

```yaml
- id: llm-pi-ai
  config:
    providers:
      litellm:
        displayName: LiteLLM proxy
        apiKeyEnv: LITELLM_API_KEY
        api: openai-completions
        baseURL: http://host:port/v1
        models:
          - id: DeepSeek-V4-flash
            contextWindow: 262144
```

对应的 `engine` 配置是 `provider: litellm`、`model: DeepSeek-V4-flash`、`env_passthrough: [LITELLM_API_KEY]`、`patches: [engine/litellm.patch.yml]`。

### 5.6 特征发现

`traj discover` 默认在 `stats` 组的特征上做 KMeans 聚类，从每个簇中取离中心最近的 trajectory，并输出渲染文件路径和所在簇的大小。
使用 `--method random` 时随机抽取。
`traj-discover` skill 指导 Claude Code 阅读这些文件，提出候选特征写入 `features/*.yaml`，运行 `traj validate` 检查，并用 `traj extract --limit` 在少量样本上试跑。
人通过 git diff 审阅后提交。

## 6. 采样

### 6.1 向量化

每个特征展开成宽表中的若干列：

| type | 查询列，用于条件筛选 | 距离列，用于聚类和离群检测 |
|---|---|---|
| scalar、boolean | `name` | `name` |
| category | `name`，字符串 | 每个标签一列 one-hot |
| set | 每个标签一列 multi-hot，以及 `name__count` | 每个标签一列 multi-hot |
| vector | `name__max`、`__min`、`__mean`、`__first`、`__last` | 重采样到 `sampling.vector_length` 个点，加上五个聚合列 |
| distribution | 每个标签一列概率 | 同左 |

只有状态为 `ok`、`spec_hash` 与当前定义一致、`sha256` 与当前渲染文件一致的特征值进入宽表。
`traj status` 按同样的条件统计每个特征组：`ok` 和 `not_ok` 是按当前定义和当前内容计算的结果，`stale` 是定义或内容变化后需要重算的数量，`missing` 是从未计算的数量。
距离列先做标准化，缺失值填为该列均值。
每个特征的权重除以其列数的平方根，使列数多的特征不会主导距离。

### 6.2 策略

```yaml
name: default
budget: 40
seed: 0
features: all
strategies:
  - kind: target
    label: frustrated
    quota: 0.25
    where: "user_frustration >= {user_frustration.high}"
    within: diversity
  - kind: outlier
    label: outlier
    quota: 0.25
    method: isolation_forest
  - kind: diversity
    label: diverse
    quota: rest
```

`features` 可以是 `all`、特征名列表，或者特征名到权重的映射。
`population` 默认为 `complete`，只在所有采样特征都已按当前定义和当前内容计算的 trajectory 中采样；值为空的特征，例如没有工具结果时的 `tool_error_rate`，也算作已计算。
`population: all` 在全部 trajectory 中采样。
`quota` 可以是 0 到 1 之间的比例、整数或 `rest`。
策略按列表顺序执行，已入选的 trajectory 不会再次入选。

| kind | 做法 | 入选原因 |
|---|---|---|
| `target` | 用 `pandas.DataFrame.query` 按 `where` 筛选，`{feature.threshold}` 替换为 YAML 中的阈值，再按 `within` 在子集内挑选 | 条件、命中数量，以及排序列的值或簇信息 |
| `outlier` | `isolation_forest` 或 `knn` 打分，取分数最高的 | 方法、分数、排名 |
| `diversity` | KMeans 聚成 quota 个簇，簇按大小依次轮流取离中心最近的未入选成员 | 簇编号、簇大小、簇占比 |
| `random` | 随机抽取 | 无 |

`within` 取值为 `diversity`、`random`、`top:<列名>` 或 `bottom:<列名>`。
采样结果写入 `selection.json` 并打印到 stdout。
结果包含采样总体的大小、每个策略的配额、实际入选数和 target 策略的命中数，以及入选列表。
入选列表中每条包含 key、渲染文件路径、策略、入选原因和该条 trajectory 的全部查询列。

## 7. 报告与反馈

`traj-report` skill 指导 Claude Code 按以下步骤工作：

1. 运行 `traj status`，确认采样用到的特征组都已按当前定义完成提取。
2. 运行 `traj table --format describe` 查看整体分布。
3. 运行 `traj sample`，得到入选列表和入选原因。
4. 阅读入选的渲染文件，核对特征值是否正确。
5. 在 `reports/<日期>-<主题>.md` 中撰写报告，每条结论引用 trajectory key 和步号。
6. 把对特征定义和采样策略的修改写入 YAML，运行 `traj validate`，与报告一起提交，commit message 写明依据的报告。

## 8. CLI

| 命令 | 作用 |
|---|---|
| `traj init <dir>` | 生成分析项目 |
| `traj ingest [dataset...]` | 读入原始数据，写出统一格式和渲染文件 |
| `traj validate [--instructions]` | 校验配置、全部特征组和采样策略，输出生成的输出 schema |
| `traj discover [--n 20] [--method diversity\|random] [--group stats]` | 挑选用于特征发现的样本 |
| `traj extract [--group g] [--dataset d] [--key k] [--limit n] [--workers n] [--dry-run]` | 提取特征 |
| `traj table [--format json\|csv\|describe] [--columns c...]` | 输出宽表 |
| `traj sample [--sampler default] [--budget n]` | 采样 |
| `traj show <key> [--cat]` | 输出某条 trajectory 的渲染文件路径和特征，或输出渲染内容 |
| `traj status` | 输出数据集规模和各特征组的提取情况 |

退出码：0 表示成功，1 表示运行错误，2 表示用法或配置错误。

## 附录

### A. 分片的用途

aifn 的 DeepSeek engine 让 agent 在 workspace 中用文件工具阅读文件，长文件可以分段读取。
分片标记用于逐片特征，保证逐片向量的长度和每个元素的含义在不同调用之间一致。

### B. 逐片向量的长度检查

输出模型按特征组生成，生成时无法得知每条 trajectory 的分片数，所以向量长度在结论写入特征表时检查。
instruction 要求向量长度等于输入中的 `n_chunks`。

### C. 测试数据

测试使用真实数据：UltraChat 和 Toucan 两个公开数据集的样本，以及本仓库设计讨论的 Claude Code session 片段。
session 片段不包含 `attachment` 记录，因为这些记录保存了用户的环境信息。
`tests/data/replay/` 保存了在三条 Toucan trajectory 上真实调用模型得到的 aifn 事件记录和提交结果，按渲染文件的内容哈希命名。
测试通过 aifn 的 `ReplayEngine` 重放这些记录，覆盖从提交任务、worker 执行、输出校验到写入特征表的完整路径。
