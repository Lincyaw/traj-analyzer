from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML

from traj_analyzer.files import read_yaml
from traj_analyzer.hunting.functions import Facet

FACETS_FILE = "facets.yaml"
PROBES_FILE = "probes.yaml"


class KeptProbe(BaseModel):
    """One probe a finished run of a hunt kept, as `hunts/<name>/probes.yaml` lists it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    facet: str
    kind: Literal["sql", "code"]
    file: str
    why: str
    hits: int
    read: int
    found: int
    precision: float | None


@dataclass(frozen=True)
class Kept:
    """The facets and the probes a finished run of a hunt kept in `hunts/<name>/`, each probe with its text."""

    facets: list[Facet]
    probes: list[tuple[KeptProbe, str]]


def read_kept(directory: Path) -> Kept | None:
    """What a finished run kept in `directory`; None before any run of the hunt finished."""
    if not (directory / PROBES_FILE).is_file():
        return None
    facets = [Facet.model_validate(facet) for facet in read_yaml(directory / FACETS_FILE)]
    entries = [KeptProbe.model_validate(entry) for entry in read_yaml(directory / PROBES_FILE)]
    return Kept(facets=facets, probes=[(entry, (directory / entry.file).read_text(encoding="utf-8"))
                                       for entry in entries])


def write_kept(directory: Path, kept: Kept) -> None:
    """Write the facets and the probes to `directory`, removing the probe files of probes it no longer keeps."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {entry.file for entry, _ in kept.probes}
    for path in [*directory.glob("*.sql"), *directory.glob("*.py")]:
        if path.name not in files:
            path.unlink()
    for entry, text in kept.probes:
        (directory / entry.file).write_text(text, encoding="utf-8")
    yaml = YAML()
    yaml.width = 4096
    yaml.dump([facet.model_dump(mode="json") for facet in kept.facets], directory / FACETS_FILE)
    yaml.dump([entry.model_dump(mode="json") for entry, _ in kept.probes], directory / PROBES_FILE)
