"""BirdNET species label locales (zh.txt from birdnet package data)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_HUMAN_ZH = {
    "Human non-vocal": "人类非语言声",
    "Human vocal": "人类语音",
    "Human whistle": "人类哨声",
}


def _split_label(label: str) -> tuple[str, str]:
    if "_" not in label:
        return label.strip(), label.strip()
    scientific, common = label.split("_", 1)
    return scientific.strip(), common.strip()


@lru_cache(maxsize=4)
def _load_common_by_scientific(locale: str) -> dict[str, str]:
    from birdnet.globals import MODEL_BACKEND_TF
    from birdnet.utils.local_data import get_lang_dir

    path = get_lang_dir("acoustic", "2.4", MODEL_BACKEND_TF) / f"{locale}.txt"
    if not path.is_file():
        return {}

    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or "_" not in line:
            continue
        scientific, common = _split_label(line)
        mapping[scientific] = common
    return mapping


def format_species_display(
    species: str,
    locale: str = "zh",
    with_scientific: bool = True,
) -> str:
    """Return localized species for terminal / GUI (default: Chinese common name)."""
    if not locale or locale.startswith("en"):
        return species

    scientific, english_common = _split_label(species)
    localized = _load_common_by_scientific(locale).get(scientific)
    if not localized or localized == english_common:
        localized = _HUMAN_ZH.get(english_common, localized or english_common)

    if with_scientific and scientific and localized != scientific:
        return f"{localized}（{scientific}）"
    return localized


def localize_prediction_rows(
    rows: list[dict[str, float | str]],
    locale: str = "zh",
) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for row in rows:
        item = dict(row)
        species = str(item.get("species", ""))
        item["species_raw"] = species
        item["species"] = format_species_display(species, locale=locale)
        out.append(item)
    return out
