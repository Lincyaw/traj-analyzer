from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from traj_analyzer.files import read_yaml

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
    description: str = ""
    """What the trajectories are and how to read them; rendered at the top of every Markdown file."""


class RenderConfig(_Strict):
    max_step_chars: int = Field(default=4000, ge=1)
    chunk_chars: int = Field(default=24000, ge=1)
    hide_metadata: list[str] = Field(default_factory=list)
    """Metadata keys, or glob patterns such as `eval_*`, left out of the rendered Markdown for judgment operators."""


class EngineConfig(_Strict):
    dsh_home: str = "~/.dsh"
    model: str = "deepseek-v4-flash"
    provider: str = "deepseek-official"
    max_tokens: int = 32768
    reasoning_effort: str | None = None
    env_passthrough: list[str] = Field(default_factory=lambda: ["DEEPSEEK_API_KEY"])
    patches: list[str] = Field(default_factory=list)


class ExtractConfig(_Strict):
    workers: int = Field(default=4, ge=1)
    """aifn worker processes for judgment groups."""
    code_workers: int = Field(default=8, ge=1)
    """Processes that load trajectories, run the parsing layers and compute code groups."""
    timeout_s: float = Field(default=900.0, gt=0)
    attempts: int = Field(default=2, ge=1)


class SamplingConfig(_Strict):
    vector_length: int = Field(default=10, ge=2)
    path_support: float = Field(default=0.01, gt=0, le=1)
    """Share of trajectories that must reach a node of a paths feature for the node to get a column."""


class ReferenceConfig(_Strict):
    datasets: Literal["all"] | list[str] = "all"
    """Datasets whose trajectories are the reference set; every other trajectory is monitored."""
    exclude: list[str] = Field(default_factory=list)
    """Trajectory keys left out of the reference set."""


class EvaluationConfig(_Strict):
    size: int = Field(default=100, ge=1)
    """Reference trajectories measured a second time."""
    seed: int = 0
    model: str | None = None
    """Model of the second measurement; it must differ from the model of every judgment group."""
    noise_max: float = Field(default=0.1, ge=0, le=1)
    dependence_min: float = Field(default=0.9, ge=0, le=1)
    label_limit: int = Field(default=100, ge=1)


class MiningConfig(_Strict):
    starting: int = Field(default=12, ge=1)
    """Trajectories that differ most under the enabled features, listed for the survey to start from."""
    hypotheses: int = Field(default=12, ge=1)
    """Most hypotheses one round tests."""
    candidates: int = Field(default=15, ge=1)
    """Most candidates one round carries into the checks."""
    trial_size: int = Field(default=30, ge=2)
    """Trajectories of its scope a judged candidate is measured on, by both models."""
    varying_min: float = Field(default=0.02, ge=0, le=1)
    """Least share of trajectories whose value differs from a candidate's most common value."""
    context_share_max: float = Field(default=0.9, gt=0, le=1)
    """A candidate is rejected when this share of its varying trajectories has one value of a context feature
    that fewer than half of the trajectories have."""
    batch_explained_min: float = Field(default=0.02, ge=0, le=1)
    """With batches declared: least share of a candidate's variance that the batches explain inside the strata."""
    extract: bool = True
    """Whether a round extracts the mined judgment features it enables from every trajectory."""
    capacity: int = Field(default=30, ge=1)
    """Most mined judgment features."""
    revise_cases: int = Field(default=10, ge=1)
    model: str | None = None
    """Model of the mine-* functions; defaults to engine.model."""
    timeout_s: float = Field(default=3600.0, gt=0)


class AlertsConfig(_Strict):
    min_batch: int = Field(default=30, ge=2)
    """Fewest trajectories a batch needs in a stratum for the stratum to be compared with the baseline."""
    min_tail_precision: float = Field(default=0.6, ge=0, le=1)
    """Share of a feature's tail that must be worth a look for the feature to raise alerts."""
    min_lift: float = Field(default=1.5, ge=0)
    """Least consequence lift of an event rule for the rule to raise alerts."""
    tail_size: int = Field(default=12, ge=1)
    """Trajectories of the tail the `mine-tail-check` function looks at per candidate."""
    min_human_verdicts: int = Field(default=5, ge=1)
    """Verdicts a person has given on a feature from which their precision replaces the model's tail precision."""
    consequences: list[str] = Field(default_factory=lambda: ["reaction", "interrupt"])
    """What a trajectory itself shows went wrong, following a turn: event rules whose event at the start of the
    next message the user typed counts, such as `reaction`, and `interrupt`, the user interrupting the turn."""


class LabelsConfig(_Strict):
    """Labels of the trajectories, used only to check the method where a dataset has them."""

    outcome: str | None = None
    """Feature column that tells how the trajectory ended: a boolean, or a scalar where low is bad."""
    failure_below: float | None = None
    """For a scalar outcome, the value below which a trajectory counts as failed; the median by default."""
    correction: str | None = None
    """Feature column that tells whether the user pushed back: a boolean, or a scalar where positive means yes."""
    turn_correction: str | None = None
    """Metadata key of a list of `{step, label}` objects, one per message the user typed, with `label` one of
    `correction`, `rejection`, `failure_report` and `non_pushback`; the judged reactions are checked against it."""


class OperatorUse(_Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    use: str
    alias: str | None = Field(default=None, alias="as")
    """Instance name; defaults to the operator name."""
    prefix: str | None = None
    """Prepended as `<prefix>_` to every feature name, to enable one operator several times."""
    call: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    tail_precision: float | None = Field(default=None, ge=0, le=1)
    """Share of the feature's tail the tail check found worth a look, written by mining."""
    lift: float | None = Field(default=None, ge=0)
    """Consequence lift of an event feature, written by mining."""
    alert: bool = False
    """Whether the feature raises alerts, from its tail precision and lift."""


class SampleConfig(_Strict):
    per_batch: int = Field(ge=1)
    """Trajectories drawn at random from every batch; the group is extracted from these alone."""
    seed: int = 0


class WindowConfig(_Strict):
    steps: int = Field(default=40, ge=2)
    stride: int = Field(default=20, ge=1)


class CallConfig(_Strict):
    model: str | None = None
    evidence: bool = True
    sample: SampleConfig | None = None
    window: WindowConfig = Field(default_factory=WindowConfig)
    """Size and stride of the windows of a call group whose operators judge windows."""
    context: int = Field(default=8, ge=0)
    """Steps before each message the user typed that a call group over turns shows, with the message itself."""
    turns_per_call: int = Field(default=20, ge=1)
    """Most turns one call of a group over turns judges; a trajectory with more turns takes several calls."""
    tail: int | None = Field(default=None, ge=1)
    """Steps from the end that a call group over whole trajectories shows, after the last message the user typed;
    the whole trajectory when unset."""


class Config(_Strict):
    version: Literal[1] = 1
    datasets: dict[str, DatasetConfig] = Field(default_factory=dict)
    layers: list[str] = Field(default_factory=list)
    """Parsing layers run on every trajectory besides those the enabled operators need."""
    operators: list[OperatorUse] = Field(default_factory=list)
    calls: dict[str, CallConfig] = Field(default_factory=dict)
    batches: list[str] = Field(default_factory=list)
    """Columns whose values make a batch: metadata as `meta.<key>`, or a feature; a scalar is cut into quartiles."""
    strata: list[str] = Field(default_factory=list)
    """Columns held fixed when batches are compared and trajectories scored, written like `batches`."""
    baseline: str = "all"
    """The batch every batch is compared with, or `all` for every other batch."""
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mining: MiningConfig = Field(default_factory=MiningConfig)


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

    def trajectory_path(self, key: str, suffix: Literal[".json", ".md"]) -> Path:
        """File of one trajectory, `<dataset>/<id>` → `.traj/datasets/<dataset>/<id><suffix>`."""
        dataset, _, tid = key.partition("/")
        return self.dataset_dir(dataset) / f"{tid}{suffix}"

    @property
    def operators_dir(self) -> Path:
        return self.root / "operators"

    @property
    def samplers_dir(self) -> Path:
        return self.root / "samplers"

    @property
    def reviews_path(self) -> Path:
        """The verdicts people gave on trajectories a feature pointed them to, one JSON line each."""
        return self.root / "reviews" / "verdicts.jsonl"

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
    def tables_dir(self) -> Path:
        """The tables of the parsing layers, the evidence, the events, the judgments, the shifts and the anomalies."""
        return self.data_dir / "tables"

    @property
    def windows_dir(self) -> Path:
        """The rendered windows of call groups whose operators judge windows."""
        return self.data_dir / "windows"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"

    @property
    def evaluation_dir(self) -> Path:
        return self.data_dir / "evaluation"

    @property
    def mining_dir(self) -> Path:
        return self.data_dir / "mining"

    @property
    def scenario_path(self) -> Path:
        """What the trajectories are, as the last finished mining round described it."""
        return self.mining_dir / "scenario.md"

    @property
    def trial_path(self) -> Path:
        """Definitions of the judged candidates under trial, in the parameters of the `llm.features` operator."""
        return self.mining_dir / "trial.json"
