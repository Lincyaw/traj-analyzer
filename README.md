# traj-analyzer

traj-analyzer 对一批任意格式的 LLM trajectory 做批量分析。
它按可配置的特征把每条 trajectory 转换成数值，采样出最值得阅读的少量样本，再由 Claude Code 阅读样本并撰写报告。
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
traj extract --group stats
traj discover --n 20
traj validate
traj extract --group outcome --limit 5
traj sample
```

`traj init` 生成的项目包含 `traj-discover` 和 `traj-report` 两个 skill。
在项目目录中启动 Claude Code 后，用前者提议特征，用后者撰写报告。

所有命令都向 stdout 输出 JSON。
退出码：0 表示成功，1 表示运行错误，2 表示用法或配置错误。

## 开发

```sh
uv run pytest
uv run ruff check src tests
```
