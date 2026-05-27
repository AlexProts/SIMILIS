from __future__ import annotations

from typing import Dict, Optional


DEFAULT_CONFIDENCE_THRESHOLDS = {
    "name_norm": 0.40,
    "material_norm": 0.50,
    "part_norm": 0.50,
    "fragm_norm": 0.30,
}

MATERIAL_ADJECTIVES = {
    "керамика": "керамический",
    "фарфор": "фарфоровый",
    "фаянс": "фаянсовый",
    "стекло": "стеклянный",
    "другое": None,
}


def material_phrase(material: str) -> Optional[str]:
    return MATERIAL_ADJECTIVES.get(material, material if material != "другое" else None)


def build_auto_description(
    pred_name: str,
    conf_name: float,
    pred_material: str,
    conf_material: float,
    pred_part: str,
    conf_part: float,
    pred_fragm: str,
    conf_fragm: float,
    thresholds: Dict[str, float] | None = None,
) -> str:
    thresholds = thresholds or DEFAULT_CONFIDENCE_THRESHOLDS
    parts = []

    if conf_name >= thresholds["name_norm"]:
        parts.append(pred_name)
    else:
        parts.append("предмет")

    mat = material_phrase(pred_material)
    if mat and conf_material >= thresholds["material_norm"]:
        parts.append(mat)

    if (
        conf_part >= thresholds["part_norm"]
        and pred_part not in {"не_указано", "другое"}
    ):
        parts.append(pred_part)

    if conf_fragm >= thresholds["fragm_norm"]:
        parts.append(pred_fragm)

    return ", ".join(parts)
