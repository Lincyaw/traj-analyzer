from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from scipy.stats import binomtest

from traj_analyzer.hunting.functions import Facet, Hit
from traj_analyzer.hunting.spec import HuntSpec

Source = Literal["hit", "coverage", "random", "final"]
"""Why a trajectory was read: a probe hit it, it was drawn from the least read cluster, it was drawn at random from
the trajectories no probe hits, or it was left over from the rounds and read by the closing check."""
CONFIDENCE = 0.95


class ProbeState(BaseModel):
    name: str
    facet: str
    kind: Literal["sql", "code"]
    text: str
    why: str
    since: int
    """The round from which the probe has this text."""
    status: Literal["active", "retired", "failed", "unproductive"] = "active"
    """`retired` when a proposal left the probe out, `failed` when it did not run, `unproductive` when the ledger
    retired it for finding nothing real; a proposal that returns an unproductive probe unchanged does not revive it."""
    reason: str = ""
    """Why the probe is not active."""


class Reading(BaseModel):
    """What one read of a trajectory found."""

    key: str
    round: int
    source: Source
    hits: list[Hit] = Field(default_factory=list)
    """Where the active probes hit the trajectory in the round it was read."""
    found: bool
    facet: str | None = None
    steps: list[int] = Field(default_factory=list)
    why: str


class RoundRecord(BaseModel):
    round: int
    reads: dict[str, int] = Field(default_factory=dict)
    """Trajectories read, by why they were drawn."""
    unanswered: list[str] = Field(default_factory=list)
    """Trajectories drawn whose case got no verdict; they stay unread."""
    failed: list[str] = Field(default_factory=list)
    """Probes that failed to run."""
    retired: list[str] = Field(default_factory=list)
    """Probes the ledger retired before the draw."""
    found: int = 0
    new: list[str] = Field(default_factory=list)
    """The new findings: a facet confirmed for the first time, or a trajectory no probe hits."""
    quiet: int = 0
    """Rounds in a row without a new finding, this one included."""


class HuntState(BaseModel):
    """Everything a hunt knows after a round; the next round starts from it alone."""

    round: int = 0
    quiet: int = 0
    facets: list[Facet] = Field(default_factory=list)
    probes: list[ProbeState] = Field(default_factory=list)
    hits: dict[str, list[str]] = Field(default_factory=dict)
    """Per active probe, the trajectories it hit in the last round."""
    readings: dict[str, Reading] = Field(default_factory=dict)
    """Per trajectory read, what was found."""
    rounds: list[RoundRecord] = Field(default_factory=list)

    def active(self) -> list[ProbeState]:
        return [probe for probe in self.probes if probe.status == "active"]

    def hit_keys(self) -> set[str]:
        return {key for probe in self.active() for key in self.hits.get(probe.name, [])}

    def found_facets(self) -> set[str]:
        return {reading.facet for reading in self.readings.values() if reading.found and reading.facet}


def probe_ledger(state: HuntState, spec: HuntSpec) -> list[dict[str, Any]]:
    """Per probe: its hits, how many were read and confirmed, its precision with its Wilson interval, its unread hits
    and how many of them are expected to be real, and what the ledger asks of it."""
    rows = []
    counts: dict[str, int] = {}
    for probe in state.active():
        for key in state.hits.get(probe.name, []):
            counts[key] = counts.get(key, 0) + 1
    for probe in state.probes:
        hits = state.hits.get(probe.name, []) if probe.status == "active" else []
        read = [key for key in hits if key in state.readings]
        found = [key for key in read if state.readings[key].found]
        precision = len(found) / len(read) if read else None
        interval = None
        if read:
            bounds = binomtest(len(found), len(read)).proportion_ci(confidence_level=CONFIDENCE, method="wilson")
            interval = [round(float(bounds.low), 4), round(float(bounds.high), 4)]
        unread = len(hits) - len(read)
        rows.append({
            "name": probe.name, "facet": probe.facet, "kind": probe.kind, "status": probe.status,
            "reason": probe.reason, "since": probe.since, "hits": len(hits), "read": len(read), "found": len(found),
            "precision": None if precision is None else round(precision, 4), "interval": interval,
            "unread": unread, "expected_unverified": None if precision is None else round(unread * precision, 2),
            "only_hits": sum(counts[key] == 1 for key in hits),
            "revise": precision is not None and len(read) >= spec.min_reads_per_probe
            and precision < spec.min_precision,
        })
    return rows


def unproductive(state: HuntState, spec: HuntSpec) -> dict[str, int]:
    """The active probes that found nothing real in twice the reads every probe gets, with their reads."""
    return {row["name"]: row["read"] for row in probe_ledger(state, spec)
            if row["status"] == "active" and row["read"] >= 2 * spec.min_reads_per_probe and row["found"] == 0}


def coverage(state: HuntState, keys: list[str], clusters: dict[str, int]) -> dict[str, Any]:
    """How much of the trajectories the reads cover, and how many real trajectories the probes may still miss.

    The bound rests on the random reads of trajectories no active probe hits: with `m` confirmed among `n`, the
    one-sided Clopper-Pearson bound of the share, times the unread trajectories no probe hits.
    """
    hit = state.hit_keys()
    read = set(state.readings)
    sizes: dict[int, int] = {}
    for key in keys:
        sizes[clusters[key]] = sizes.get(clusters[key], 0) + 1
    clusters_read = {clusters[key] for key in read}
    random = [reading for reading in state.readings.values() if reading.source == "random" and reading.key not in hit]
    misses = sum(reading.found for reading in random)
    upper = None
    if random:
        bounds = binomtest(misses, len(random), alternative="less").proportion_ci(confidence_level=CONFIDENCE,
                                                                                  method="exact")
        upper = float(bounds.high)
    unread_unhit = sum(1 for key in keys if key not in hit and key not in read)
    return {
        "trajectories": len(keys), "read": len(read), "read_share": round(len(read) / len(keys), 4),
        "clusters": len(sizes), "clusters_read": len(clusters_read),
        "hit": len(hit), "unread_unhit": unread_unhit,
        "random_reads_unhit": len(random), "random_misses": misses,
        "miss_share_upper": None if upper is None else round(upper, 4),
        "expected_missed": None if upper is None else round(upper * unread_unhit, 2),
    }


def ledger(state: HuntState, spec: HuntSpec, keys: list[str], clusters: dict[str, int]) -> dict[str, Any]:
    """What the next proposal reads: the probes, the facets, the rounds and the coverage."""
    facets = {facet.name: 0 for facet in state.facets}
    for reading in state.readings.values():
        if reading.found and reading.facet is not None:
            facets[reading.facet] = facets.get(reading.facet, 0) + 1
    return {
        "round": state.round, "quiet": state.quiet, "patience": spec.patience,
        "probes": probe_ledger(state, spec),
        "facets": [{"name": name, "found": count} for name, count in facets.items()],
        "rounds": [record.model_dump(mode="json") for record in state.rounds],
        "coverage": coverage(state, keys, clusters),
    }


def new_findings(state: HuntState, readings: list[Reading]) -> list[str]:
    """What the readings of a round add: every facet confirmed for the first time, and every confirmed trajectory no
    probe hits. `state` is the state before the round."""
    seen = state.found_facets()
    found = []
    for reading in readings:
        if not reading.found:
            continue
        assert reading.facet is not None
        if reading.facet not in seen:
            found.append(f"facet {reading.facet} first confirmed in {reading.key} at steps {reading.steps}")
            seen.add(reading.facet)
        if reading.source in ("coverage", "random"):
            found.append(f"{reading.key} shows facet {reading.facet} at steps {reading.steps} and no probe hits it")
    return found
