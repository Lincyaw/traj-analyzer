from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from traj_analyzer.hunting.ledger import Reading
from traj_analyzer.layers.tables import column, table_path, with_key
from traj_analyzer.project import Project

VERDICTS_TABLE = "hunt_verdicts"
VERDICTS_SCHEMA = pa.schema([
    column("hunt", pa.string(), "The hunt that read the trajectory, `hunts/<hunt>.yaml`."),
    column("found", pa.bool_(), "Whether the trajectory shows what the hunt's need describes."),
    column("facet", pa.string(), "When found, the kind of behaviour of the need it shows."),
    column("step", pa.int64(), "When found, the first step where it shows."),
    column("steps", pa.list_(pa.int64()), "When found, every step where it shows, as the verdict cites them."),
    column("why", pa.string(), "What the verdict saw, naming the steps."),
    column("round", pa.int64(), "The round that read the trajectory; the round after the last for the closing "
                                "check."),
    column("source", pa.string(), "Why it was read: hit, a probe hit it; coverage, drawn from the cluster read "
                                  "least; random, drawn at random from the trajectories no probe hits; final, read by "
                                  "the closing check."),
    column("probes", pa.list_(pa.string()), "The probes that hit the trajectory when it was read."),
])


def write_verdicts(project: Project, name: str, readings: list[Reading]) -> None:
    """Replace the hunt's rows of the verdicts table with every trajectory it read."""
    rows = [{"key": reading.key, "hunt": name, "found": reading.found, "facet": reading.facet,
             "step": min(reading.steps) if reading.steps else None, "steps": reading.steps, "why": reading.why,
             "round": reading.round, "source": reading.source,
             "probes": sorted({hit.probe for hit in reading.hits})}
            for reading in sorted(readings, key=lambda r: r.key)]
    schema = with_key(VERDICTS_SCHEMA)
    table = pa.Table.from_pylist(rows, schema=schema)
    path = table_path(project, VERDICTS_TABLE)
    if path.is_file():
        kept = pq.read_table(path, schema=schema)
        table = pa.concat_tables([kept.filter(pc.not_equal(kept["hunt"], name)), table])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
