from __future__ import annotations

import csv
import re
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )


def resolve_evidence(value: str | None, filenames: list[str]) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    search_roots = [
        Path.cwd(),
        REPO_ROOT,
        REPO_ROOT / "release-assets",
        REPO_ROOT / "release-assets" / "anonymous",
    ]
    for root in search_roots:
        for filename in filenames:
            candidate = root / filename
            if candidate.exists():
                return candidate.resolve()
    names = ", ".join(filenames)
    raise FileNotFoundError(
        f"Evidence not found. Pass an extracted bundle or archive for one of: {names}"
    )


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(f"Archive links are not supported: {member.name}")
        target = (destination / member.name).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe archive member: {member.name}")
    archive.extractall(destination)


@contextmanager
def materialize_evidence(path: Path):
    path = path.resolve()
    if path.is_dir():
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="dicra_figure_evidence_") as tmp:
        destination = Path(tmp)
        with tarfile.open(path, "r:gz") as archive:
            _safe_extract(archive, destination)
        children = [child for child in destination.iterdir() if child.is_dir()]
        yield children[0] if len(children) == 1 else destination


def find_file(root: Path, filename: str, contains: str | None = None) -> Path:
    matches = sorted(root.rglob(filename))
    if contains:
        token = contains.replace("\\", "/")
        matches = [path for path in matches if token in path.as_posix()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {filename} under {root}, found {len(matches)}: {matches}"
        )
    return matches[0]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def keyed_rows(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in result:
            raise ValueError(f"Duplicate row key: {key}")
        result[key] = row
    return result


INTERVAL_PATTERN = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*\]\s*$"
)
MEAN_STD_PATTERN = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*\+/-\s*(-?\d+(?:\.\d+)?)\s*$"
)


def parse_interval(value: str) -> tuple[float, float, float]:
    match = INTERVAL_PATTERN.match(value)
    if not match:
        raise ValueError(f"Invalid interval: {value!r}")
    return tuple(float(item) for item in match.groups())


def parse_mean_std(value: str) -> tuple[float, float]:
    match = MEAN_STD_PATTERN.match(value)
    if not match:
        raise ValueError(f"Invalid mean/std value: {value!r}")
    return tuple(float(item) for item in match.groups())


def ensure_keys(actual, expected, label: str) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    if actual_set != expected_set:
        raise ValueError(
            f"{label} mismatch: missing={sorted(expected_set - actual_set)}, "
            f"extra={sorted(actual_set - expected_set)}"
        )


def save_figure(fig, output: str, png: str | None = None) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "Python/Matplotlib"},
    )
    if png:
        png_path = Path(png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    print(f"wrote {output_path.resolve()}")
