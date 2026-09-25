from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

CONFIG_FILE = "traj.yaml"


class ConfigError(Exception):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(_Strict):
    adapter: str
    input: list[str]
    exclude: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class RenderConfig(_Strict):
    max_step_chars: int = Field(default=4000, ge=1)
    chunk_chars: int = Field(default=24000, ge=1)
    hide_metadata: list[str] = Field(default_factory=list)
    """Metadata keys left out of the rendered Markdown, so that LLM operators do not see them."""


class EngineConfig(_Strict):
    kind: Literal["deepseek"] = "deepseek"
    dsh_home: str = "~/.dsh"
    model: str = "deepseek-v4-flash"
    provider: str = "deepseek-official"
    max_tokens: int = 32768
    reasoning_effort: str | None = None
    env_passthrough: list[str] = Field(default_factory=lambda: ["DEEPSEEK_API_KEY"])
    patches: list[str] = Field(default_factory=list)


class ExtractConfig(_Strict):
    workers: int = Field(default=4, ge=1)
    timeout_s: float = Field(default=900.0, gt=0)
    attempts: int = Field(default=2, ge=1)


class SamplingConfig(_Strict):
    vector_length: int = Field(default=10, ge=2)


class OperatorUse(_Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    use: str
    alias: str | None = Field(default=None, alias="as")
    call: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class CallConfig(_Strict):
    model: str | None = None
    evidence: bool = True


class Config(_Strict):
    version: Literal[1] = 1
    datasets: dict[str, DatasetConfig] = Field(default_factory=dict)
    operators: list[OperatorUse] = Field(default_factory=list)
    calls: dict[str, CallConfig] = Field(default_factory=dict)
    render: RenderConfig = Field(default_factory=RenderConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class Project:
    def __init__(self, root: Path, config: Config) -> None:
        self.root = root
        self.config = config

    @classmethod
    def find(cls, start: Path | None = None) -> Project:
        # TRAJ_PROJECT lets worker processes started by `traj extract` load the same project.
        if env := os.environ.get("TRAJ_PROJECT"):
            return cls.load(Path(env))
        here = (start or Path.cwd()).resolve()
        for directory in (here, *here.parents):
            if (directory / CONFIG_FILE).is_file():
                return cls.load(directory)
        raise ConfigError(f"No {CONFIG_FILE} found in {here} or its parents")

    @classmethod
    def load(cls, root: Path) -> Project:
        root = root.expanduser().resolve()
        config = Config.model_validate(read_yaml(root / CONFIG_FILE))
        return cls(root, config)

    @property
    def data_dir(self) -> Path:
        return self.root / ".traj"

    def dataset_dir(self, dataset: str) -> Path:
        return self.data_dir / "datasets" / dataset

    @property
    def operators_dir(self) -> Path:
        return self.root / "operators"

    @property
    def samplers_dir(self) -> Path:
        return self.root / "samplers"

    @property
    def instructions_dir(self) -> Path:
        return self.data_dir / "instructions"

    @property
    def mailbox_dir(self) -> Path:
        return self.data_dir / "mailbox"

    @property
    def table_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"
