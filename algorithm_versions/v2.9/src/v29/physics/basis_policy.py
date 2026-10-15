"""Explicit basis policy prevents accidental double counting of mutual coupling."""

from __future__ import annotations

from enum import Enum


class BasisType(str, Enum):
    IDEAL_PHASE = "ideal_phase"
    ISOLATED_ELEMENT_PATTERN = "isolated_element_pattern"
    COMMON_PATTERN = "common_pattern"
    EMBEDDED_ELEMENT_BASIS = "embedded_element_basis"


def validate_basis_policy(basis_type: BasisType | str, *, enable_coupling: bool) -> BasisType:
    basis = BasisType(basis_type)
    if basis is BasisType.EMBEDDED_ELEMENT_BASIS and enable_coupling:
        raise ValueError(
            "embedded_element_basis already contains terminated-array coupling; "
            "an additional coupling layer would double count it"
        )
    return basis
