# Copyright (c) 2026 imec
# SPDX-License-Identifier: MIT
import re
import numbers
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
    ("K", 1e3),
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

IEC_PREFIXES = [
    ("Qi", 2**100),
    ("Ri", 2**90),
    ("Yi", 2**80),
    ("Zi", 2**70),
    ("Ei", 2**60),
    ("Pi", 2**50),
    ("Ti", 2**40),
    ("Gi", 2**30),
    ("Mi", 2**20),
    ("Ki", 2**10),
    ("ki", 2**10),
    ("", 1)
]

PREFIXES = sorted(list(set(SI_PREFIXES + IEC_PREFIXES)), key=lambda x: x[1], reverse=True)

SI_PREFIXES_MAP = {k: v for k, v in SI_PREFIXES}
IEC_PREFIXES_MAP = {k: v for k, v in IEC_PREFIXES}
PREFIXES_MAP = {k: v for k, v in PREFIXES}

unit_re = re.compile(r"^([\-+0-9.e]+)\s*({})?([a-zA-Z/^]*)$".format("|".join([x[0] for x in PREFIXES if x[0]])))

def unit_str_to_float(s, unit=None, prefixes_map=PREFIXES_MAP):
    if isinstance(s, numbers.Real):
        return float(s)

    must_specify_unit = False
    if unit:
        if isinstance(unit, str):
            unit = unit.strip()
            must_specify_unit = unit and unit in prefixes_map
        elif isinstance(unit, dict):
            must_specify_unit = any([x in prefixes_map for x in unit])
        else:
            raise ValueError("Invalid type for unit (should be string or dict) - '{}'".format(type(unit).__name__))

    match = unit_re.match(s.strip())
    if not match:
        raise ValueError("'{}' is not a valid unit string".format(s))

    value = float(match.group(1))
    prefix = match.group(2) or ""
    match_unit = match.group(3)
    if not match_unit and (must_specify_unit or prefix not in prefixes_map):
        match_unit = prefix
        prefix = ""

    unit_mult = 1
    if match_unit:
        allowed_units = unit if unit else ""
        unit_okay = True
        if not unit:
            unit_okay = False
        elif isinstance(unit, dict):
            if match_unit in unit:
                unit_mult = unit[match_unit]
            else:
                unit_okay = False
                allowed_units = list(unit.keys())
        elif isinstance(unit, str):
            if match_unit != unit:
                unit_okay = False

        if not unit_okay:
            raise ValueError("For '{}' unit ('{}') does not match specified '{}'".format(s, match_unit, allowed_units))

    if prefix not in prefixes_map:
        raise ValueError("Invalid prefix '{}' - choose from '{}'".format(prefix, list(prefixes_map.keys())))

    return value * prefixes_map[prefix] * unit_mult

@lru_cache
def s_to_float(s):
    if isinstance(s, numbers.Real):
        return float(s)
    return unit_str_to_float(s, "s", SI_PREFIXES_MAP)

@lru_cache
def time_to_float(s):
    if isinstance(s, numbers.Real):
        return float(s)
    return unit_str_to_float(s, {"s": 1, "m": 60, "h": 60*60, "d": 60*60*24}, SI_PREFIXES_MAP)

@lru_cache
def bit_to_float(s):
    if isinstance(s, numbers.Real):
        return float(s)
    return unit_str_to_float(s, {"b": 1, "B": 8})

@lru_cache
def byte_to_float(s):
    if isinstance(s, numbers.Real):
        return float(s)
    return unit_str_to_float(s, {"b": 1/8, "B": 1})

@lru_cache
def bps_to_float(s):
    if isinstance(s, numbers.Real):
        return float(s)
    return unit_str_to_float(s, {"bps": 1, "b/s": 1, "Bps": 8, "B/s": 8})

@lru_cache
def byte_per_s_to_float(s):
    if isinstance(s, numbers.Real):
        return float(s)
    return unit_str_to_float(s, {"bps": 1/8, "b/s": 1/8, "Bps": 1, "B/s": 1})

