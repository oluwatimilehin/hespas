# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import re
from functools import lru_cache

SI_PREFIXES = [
    ("Q", 1e30),
    ("R", 1e27),
    ("Y", 1e24),
    ("Z", 1e21),
    ("E", 1e18),
    ("P", 1e15),
    ("T", 1e12),
    ("G", 1e9),
    ("M", 1e6),
    ("k", 1e3),
    ("", 1),
    ("m", 1e-3),
    ("u", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
    ("a", 1e-18),
    ("z", 1e-21),
    ("y", 1e-24),
    ("r", 1e-27),
    ("q", 1e-30)
]

SI_PREFIXES_MAP = {k: v for k, v in SI_PREFIXES}

unit_re = re.compile(r"([\-+0-9.e]+)\s*([a-zA-Z]?)([a-zA-Z]*)")

@lru_cache
def unit_str_to_float(s, unit):
    unit = unit.strip()
    match = unit_re.match(s.strip())
    if not match:
        raise ValueError("'{}' is not a valid unit string".format(s))
    value = float(match.group(1))
    prefix = match.group(2) if match.group(2) in SI_PREFIXES_MAP else None
    match_unit = match.group(2) if prefix is None else match.group(3)
    if match_unit and match_unit != unit:
        raise ValueError("For '{}' unit ('{}') does not match specified '{}'".format(s, match_unit, unit))
    return value * SI_PREFIXES_MAP[prefix if prefix is not None else ""]

@lru_cache
def s_to_float(s):
    if isinstance(s, (int, float)):
        return float(s)
    return unit_str_to_float(s, "s")
