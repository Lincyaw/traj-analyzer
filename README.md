# traj-analyzer

traj-analyzer 对一批任意格式的 LLM trajectory 做批量分析。
它用可以选择启用的算子把每条 trajectory 转换成数值特征，采样出最值得阅读的少量样本，再由 Claude Code 阅读样本并撰写报告。
一个算子是一个 Python 文件，内置算子库在 `src/traj_analyzer/operators/library/`，分析项目可以在自己的 `operators/` 中添加算子。
设计见 [docs/design.md](docs/design.md)。

## 安装

```sh
uv sync
uv run python -m aifn install --dsh-home ~/.dsh
export DEEPSEEK_API_KEY=...
```

第二条命令把 aifn 的 harness bundle 安装到 dsh home，LLM 特征提取通过它执行。

## 使用

```sh
traj init ~/analysis/cc-sessions
cd ~/analysis/cc-sessions && git init
traj ingest
traj operators list
traj extract --group stats-basic --group stats-tool_usage
traj discover --n 20
traj operators enable collab.user_frustration --call outcome
traj validate
traj extract --group outcome --limit 5
traj sample
```

`traj init` 生成的 `traj.yaml` 已经启用了内置算子库中的全部算子，四个 llm 算子共用 `outcome` 调用组。
项目还包含三个 skill：`traj-features` 说明特征怎样设定和提取，`traj-discover` 用来阅读样本并启用或编写算子，`traj-report` 用来撰写报告。

所有命令都向 stdout 输出 JSON。
退出码：0 表示成功，1 表示运行错误，2 表示用法或配置错误。

## 开发

```sh
uv run pytest
uv run ruff check src tests
```
