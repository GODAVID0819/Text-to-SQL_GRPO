import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set


MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

DAYS = [
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
    "FRIDAY", "SATURDAY", "SUNDAY"
]

AIRLINES = [
    "AA", "UA", "DL", "WN", "AS", "B6", "NK", "F9"
]

AIRLINE_CODE_TO_NAMES: Dict[str, List[str]] = {
    "AA": ["aa", "american", "american airlines"],
    "UA": ["ua", "united", "united airlines"],
    "DL": ["dl", "delta", "delta airlines"],
    "WN": ["wn", "southwest", "southwest airlines"],
    "AS": ["as", "alaska", "alaska airlines"],
    "B6": ["b6", "jetblue", "jet blue"],
    "NK": ["nk", "spirit", "spirit airlines"],
    "F9": ["f9", "frontier", "frontier airlines"],
}

CITIES = [
    "ATLANTA", "BALTIMORE", "BOSTON", "CHICAGO", "DALLAS", "DENVER",
    "DETROIT", "HOUSTON", "LAS VEGAS", "LOS ANGELES", "MIAMI",
    "MINNEAPOLIS", "NASHVILLE", "NEW YORK", "NEWARK", "OAKLAND",
    "ORLANDO", "PHILADELPHIA", "PHOENIX", "PITTSBURGH", "SAN DIEGO",
    "SAN FRANCISCO", "SAN JOSE", "SEATTLE", "ST. LOUIS", "TAMPA",
    "WASHINGTON"
]

TIME_BUCKETS = [
    ("before noon", "<", 1200),
    ("after noon", ">", 1200),
    ("before 5 pm", "<", 1700),
    ("after 5 pm", ">", 1700),
    ("before 8 am", "<", 800),
    ("after 8 pm", ">", 2000),
]

OP_VAL_TO_PHRASE = {(op, val): phrase for phrase, op, val in TIME_BUCKETS}
TIME_PHRASES = [phrase for phrase, _, _ in TIME_BUCKETS]

ORDINALS = {
    1: ["1", "first", "the first"],
    2: ["2", "second", "the second"],
    3: ["3", "third", "the third"],
    4: ["4", "fourth", "the fourth"],
    5: ["5", "fifth", "the fifth"],
    6: ["6", "sixth", "the sixth"],
    7: ["7", "seventh", "the seventh"],
    8: ["8", "eighth", "the eighth"],
    9: ["9", "ninth", "the ninth"],
    10: ["10", "tenth", "the tenth"],
    11: ["11", "eleventh", "the eleventh"],
    12: ["12", "twelfth", "the twelfth"],
    13: ["13", "thirteenth", "the thirteenth"],
    14: ["14", "fourteenth", "the fourteenth"],
    15: ["15", "fifteenth", "the fifteenth"],
    16: ["16", "sixteenth", "the sixteenth"],
    17: ["17", "seventeenth", "the seventeenth"],
    18: ["18", "eighteenth", "the eighteenth"],
    19: ["19", "nineteenth", "the nineteenth"],
    20: ["20", "twentieth", "the twentieth"],
    21: ["21", "twenty first", "twenty-first", "the twenty first", "the twenty-first"],
    22: ["22", "twenty second", "twenty-second", "the twenty second", "the twenty-second"],
    23: ["23", "twenty third", "twenty-third", "the twenty third", "the twenty-third"],
    24: ["24", "twenty fourth", "twenty-fourth", "the twenty fourth", "the twenty-fourth"],
    25: ["25", "twenty fifth", "twenty-fifth", "the twenty fifth", "the twenty-fifth"],
    26: ["26", "twenty sixth", "twenty-sixth", "the twenty sixth", "the twenty-sixth"],
    27: ["27", "twenty seventh", "twenty-seventh", "the twenty seventh", "the twenty-seventh"],
    28: ["28", "twenty eighth", "twenty-eighth", "the twenty eighth", "the twenty-eighth"],
}



# Basic file utilities


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: Path, lines: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()



# Regex helpers


def phrase_pattern(phrase: str) -> str:
    """
    Build a loose word-boundary pattern for a phrase.

    Example:
        "san francisco" -> matches "san francisco"
        "st. louis" -> matches "st. louis" or "st louis"
    """
    phrase = phrase.lower().strip()
    phrase = phrase.replace(".", r"\.?")
    phrase = re.sub(r"\s+", r"\\s+", phrase)
    return rf"(?<![a-z0-9]){phrase}(?![a-z0-9])"


def contains_phrase(text: str, phrase: str) -> bool:
    return re.search(phrase_pattern(phrase), text.lower(), flags=re.IGNORECASE) is not None


def replace_phrase_once_or_all(text: str, old: str, new: str) -> Tuple[str, bool]:
    """
    Replace all case-insensitive occurrences of old phrase.
    Returns (new_text, did_replace).
    """
    pattern = phrase_pattern(old)
    new_text, n = re.subn(pattern, new.lower(), text, flags=re.IGNORECASE)
    return new_text, n > 0


def replace_required(text: str, old: str, new: str) -> Optional[str]:
    """
    Replace old with new in NL. If old is not found, return None.
    """
    out, did = replace_phrase_once_or_all(text, old, new)
    if not did:
        return None
    return out



# SQL extraction and replacement


def extract_city_alias_values(sql: str, alias: str) -> List[str]:
    pattern = rf"{re.escape(alias)}\.city_name\s*=\s*'([^']+)'"
    return re.findall(pattern, sql)


def replace_city_alias(sql: str, alias: str, new_values: List[str]) -> str:
    pattern = rf"({re.escape(alias)}\.city_name\s*=\s*')([^']+)(')"
    idx = 0

    def repl(match):
        nonlocal idx
        if idx >= len(new_values):
            return match.group(0)
        out = match.group(1) + new_values[idx] + match.group(3)
        idx += 1
        return out

    return re.sub(pattern, repl, sql)


def extract_airlines(sql: str) -> List[str]:
    return re.findall(r"flight_\d+\.airline_code\s*=\s*'([^']+)'", sql)


def replace_airlines(sql: str, new_values: List[str]) -> str:
    pattern = r"(flight_\d+\.airline_code\s*=\s*')([^']+)(')"
    idx = 0

    def repl(match):
        nonlocal idx
        if idx >= len(new_values):
            return match.group(0)
        out = match.group(1) + new_values[idx] + match.group(3)
        idx += 1
        return out

    return re.sub(pattern, repl, sql)


def extract_day_names(sql: str) -> List[str]:
    return re.findall(r"days_\d+\.day_name\s*=\s*'([^']+)'", sql)


def replace_day_names(sql: str, new_values: List[str]) -> str:
    pattern = r"(days_\d+\.day_name\s*=\s*')([^']+)(')"
    idx = 0

    def repl(match):
        nonlocal idx
        if idx >= len(new_values):
            return match.group(0)
        out = match.group(1) + new_values[idx] + match.group(3)
        idx += 1
        return out

    return re.sub(pattern, repl, sql)


def extract_date(sql: str) -> Optional[Tuple[int, int, int]]:
    y = re.search(r"date_day_\d+\.year\s*=\s*(\d+)", sql)
    m = re.search(r"date_day_\d+\.month_number\s*=\s*(\d+)", sql)
    d = re.search(r"date_day_\d+\.day_number\s*=\s*(\d+)", sql)
    if y and m and d:
        return int(y.group(1)), int(m.group(1)), int(d.group(1))
    return None


def replace_date(sql: str, year: int, month: int, day: int) -> str:
    sql = re.sub(r"(date_day_\d+\.year\s*=\s*)\d+", rf"\g<1>{year}", sql)
    sql = re.sub(r"(date_day_\d+\.month_number\s*=\s*)\d+", rf"\g<1>{month}", sql)
    sql = re.sub(r"(date_day_\d+\.day_number\s*=\s*)\d+", rf"\g<1>{day}", sql)
    return sql


def extract_time_constraints(sql: str, field: str) -> List[Tuple[str, int]]:
    pattern = rf"flight_\d+\.{field}\s*([<>]=?)\s*(\d+)"
    return [(op, int(v)) for op, v in re.findall(pattern, sql)]


def replace_time_constraints(sql: str, field: str, new_constraints: List[Tuple[str, int]]) -> str:
    pattern = rf"(flight_\d+\.{field}\s*)([<>]=?)(\s*)(\d+)"
    idx = 0

    def repl(match):
        nonlocal idx
        if idx >= len(new_constraints):
            return match.group(0)
        op, val = new_constraints[idx]
        idx += 1
        return f"{match.group(1)}{op}{match.group(3)}{val}"

    return re.sub(pattern, repl, sql)



# Safe sampling


def pick_distinct_strict(pool: List[str], k: int, avoid: Optional[Set[str]] = None) -> Optional[List[str]]:
    """
    Strict version:
    - never violates avoid
    - never samples duplicates
    - returns None if impossible
    """
    avoid = avoid or set()
    choices = [x for x in pool if x not in avoid]

    if len(choices) < k:
        return None

    return random.sample(choices, k)


def sample_new_date(old_date: Tuple[int, int, int]) -> Tuple[int, int, int]:
    year = old_date[0]
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return year, month, day



# NL slot phrase variants


def city_to_nl_variants(city: str) -> List[str]:
    c = city.lower()
    variants = {c}

    if c == "st. louis":
        variants.add("st louis")
        variants.add("saint louis")

    return sorted(variants, key=len, reverse=True)


def airline_to_nl_variants(code: str) -> List[str]:
    return AIRLINE_CODE_TO_NAMES.get(code.upper(), [code.lower()])


def date_to_nl_variants(date_info: Tuple[int, int, int]) -> List[str]:
    _, month, day = date_info
    month_name = MONTHS[month - 1]

    variants = []
    for d in ORDINALS.get(day, [str(day)]):
        variants.append(f"{month_name} {d}")

    return sorted(set(variants), key=len, reverse=True)


def canonical_date_phrase(date_info: Tuple[int, int, int]) -> str:
    _, month, day = date_info
    return f"{MONTHS[month - 1]} {day}"



# NL substitution helpers


def replace_any_variant_required(text: str, old_variants: List[str], new_phrase: str) -> Optional[str]:
    """
    Replace one of the old variants in text.
    If none are found, return None.
    """
    for old in old_variants:
        out, did = replace_phrase_once_or_all(text, old, new_phrase)
        if did:
            return out
    return None


def replace_city_in_nl(text: str, old_city: str, new_city: str) -> Optional[str]:
    return replace_any_variant_required(
        text,
        city_to_nl_variants(old_city),
        new_city.lower()
    )


def replace_airline_in_nl(text: str, old_code: str, new_code: str) -> Optional[str]:
    old_variants = airline_to_nl_variants(old_code)
    new_phrase = new_code.lower()
    return replace_any_variant_required(text, old_variants, new_phrase)


def replace_day_in_nl(text: str, old_day: str, new_day: str) -> Optional[str]:
    return replace_required(text, old_day.lower(), new_day.lower())


def replace_date_in_nl(
    text: str,
    old_date: Tuple[int, int, int],
    new_date: Tuple[int, int, int],
) -> Optional[str]:
    old_variants = date_to_nl_variants(old_date)
    new_phrase = canonical_date_phrase(new_date)
    return replace_any_variant_required(text, old_variants, new_phrase)


def replace_time_phrase_in_nl(
    text: str,
    old_constraint: Tuple[str, int],
    new_constraint: Tuple[str, int],
) -> Optional[str]:
    old_phrase = OP_VAL_TO_PHRASE.get(old_constraint)
    new_phrase = OP_VAL_TO_PHRASE.get(new_constraint)

    if old_phrase is None or new_phrase is None:
        return None

    return replace_required(text, old_phrase, new_phrase)


def old_slots_still_visible(
    nl: str,
    old_cities: List[str],
    old_airlines: List[str],
    old_days: List[str],
    old_date: Optional[Tuple[int, int, int]],
) -> bool:
    low = nl.lower()

    for city in old_cities:
        for variant in city_to_nl_variants(city):
            if contains_phrase(low, variant):
                return True

    for code in old_airlines:
        for variant in airline_to_nl_variants(code):
            if contains_phrase(low, variant):
                return True

    for day in old_days:
        if contains_phrase(low, day.lower()):
            return True

    if old_date is not None:
        for variant in date_to_nl_variants(old_date):
            if contains_phrase(low, variant):
                return True

    return False



# Risk filtering


def is_risky_sql(sql: str) -> bool:
    """
    Skip examples whose semantics are likely not captured by simple slot substitution.

    You can loosen this later, but this is a safe default.
    """
    s = sql.lower()

    risky_patterns = [
        " count(",
        " min(",
        " max(",
        " avg(",
        " sum(",
        " in ( select",
        " exists ",
        " not exists ",
        " group by ",
        " having ",
    ]

    return any(p in s for p in risky_patterns)



# Main augmentation


def augment_pair(nl: str, sql: str, skip_risky: bool = True) -> Optional[Tuple[str, str]]:
    if skip_risky and is_risky_sql(sql):
        return None

    old_from = extract_city_alias_values(sql, "city_1")
    old_to = extract_city_alias_values(sql, "city_2")
    old_stop = extract_city_alias_values(sql, "city_3")
    old_airlines = extract_airlines(sql)
    old_days = extract_day_names(sql)
    old_date = extract_date(sql)
    old_arr = extract_time_constraints(sql, "arrival_time")
    old_dep = extract_time_constraints(sql, "departure_time")

    new_sql = sql
    new_nl = nl

    used_cities: Set[str] = set()

    # ---------------- cities ----------------
    if old_from:
        new_from = pick_distinct_strict(CITIES, len(old_from), avoid=used_cities)
        if new_from is None:
            return None
        used_cities.update(new_from)

        for old, new in zip(old_from, new_from):
            new_nl = replace_city_in_nl(new_nl, old, new)
            if new_nl is None:
                return None

        new_sql = replace_city_alias(new_sql, "city_1", new_from)
    else:
        new_from = []

    if old_to:
        new_to = pick_distinct_strict(CITIES, len(old_to), avoid=used_cities)
        if new_to is None:
            return None
        used_cities.update(new_to)

        for old, new in zip(old_to, new_to):
            new_nl = replace_city_in_nl(new_nl, old, new)
            if new_nl is None:
                return None

        new_sql = replace_city_alias(new_sql, "city_2", new_to)
    else:
        new_to = []

    if old_stop:
        new_stop = pick_distinct_strict(CITIES, len(old_stop), avoid=used_cities)
        if new_stop is None:
            return None
        used_cities.update(new_stop)

        for old, new in zip(old_stop, new_stop):
            new_nl = replace_city_in_nl(new_nl, old, new)
            if new_nl is None:
                return None

        new_sql = replace_city_alias(new_sql, "city_3", new_stop)
    else:
        new_stop = []

    # airlines 
    if old_airlines:
        new_airlines = pick_distinct_strict(AIRLINES, len(old_airlines))
        if new_airlines is None:
            return None

        for old, new in zip(old_airlines, new_airlines):
            new_nl = replace_airline_in_nl(new_nl, old, new)
            if new_nl is None:
                return None

        new_sql = replace_airlines(new_sql, new_airlines)
    else:
        new_airlines = []

    # date / day
    if old_date is not None:
        new_date = sample_new_date(old_date)

        new_nl = replace_date_in_nl(new_nl, old_date, new_date)
        if new_nl is None:
            return None

        new_sql = replace_date(new_sql, *new_date)
        new_days = []

    elif old_days:
        new_days = pick_distinct_strict(DAYS, len(old_days))
        if new_days is None:
            return None

        for old, new in zip(old_days, new_days):
            new_nl = replace_day_in_nl(new_nl, old, new)
            if new_nl is None:
                return None

        new_sql = replace_day_names(new_sql, new_days)
        new_date = None

    else:
        new_days = []
        new_date = None

    # times
    if old_arr:
        possible = [(op, val) for _, op, val in TIME_BUCKETS]
        new_arr = [random.choice(possible) for _ in old_arr]

        for old_c, new_c in zip(old_arr, new_arr):
            new_nl = replace_time_phrase_in_nl(new_nl, old_c, new_c)
            if new_nl is None:
                return None

        new_sql = replace_time_constraints(new_sql, "arrival_time", new_arr)
    else:
        new_arr = []

    if old_dep:
        possible = [(op, val) for _, op, val in TIME_BUCKETS]
        new_dep = [random.choice(possible) for _ in old_dep]

        for old_c, new_c in zip(old_dep, new_dep):
            new_nl = replace_time_phrase_in_nl(new_nl, old_c, new_c)
            if new_nl is None:
                return None

        new_sql = replace_time_constraints(new_sql, "departure_time", new_dep)
    else:
        new_dep = []

    new_nl = normalize_ws(new_nl)
    new_sql = normalize_ws(new_sql)

    old_cities = old_from + old_to + old_stop
    if old_slots_still_visible(
        new_nl,
        old_cities=old_cities,
        old_airlines=old_airlines,
        old_days=old_days,
        old_date=old_date,
    ):
        return None

    # Avoid exact no-op.
    if normalize_ws(nl).lower() == new_nl.lower() and normalize_ws(sql) == new_sql:
        return None

    return new_nl, new_sql


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_nl", required=True)
    parser.add_argument("--train_sql", required=True)
    parser.add_argument("--out_nl", required=True)
    parser.add_argument("--out_sql", required=True)
    parser.add_argument("--copies_per_example", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include_risky",
        action="store_true",
        help="If set, also augment COUNT/MIN/MAX/nested SQL examples. Default is to skip them."
    )
    args = parser.parse_args()

    random.seed(args.seed)

    nl_lines = read_lines(Path(args.train_nl))
    sql_lines = read_lines(Path(args.train_sql))

    if len(nl_lines) != len(sql_lines):
        raise ValueError("Mismatched number of lines between NL and SQL files.")

    out_nl = list(nl_lines)
    out_sql = list(sql_lines)

    seen = set(
        (normalize_ws(nl).lower(), normalize_ws(sql))
        for nl, sql in zip(nl_lines, sql_lines)
    )

    attempted = 0
    added = 0
    skipped = 0

    for nl, sql in zip(nl_lines, sql_lines):
        for _ in range(args.copies_per_example):
            attempted += 1

            aug = augment_pair(
                nl=nl,
                sql=sql,
                skip_risky=not args.include_risky,
            )

            if aug is None:
                skipped += 1
                continue

            aug_nl, aug_sql = aug
            key = (normalize_ws(aug_nl).lower(), normalize_ws(aug_sql))

            if key in seen:
                skipped += 1
                continue

            seen.add(key)
            out_nl.append(aug_nl)
            out_sql.append(aug_sql)
            added += 1

    write_lines(Path(args.out_nl), out_nl)
    write_lines(Path(args.out_sql), out_sql)

    print(f"Original: {len(nl_lines)}")
    print(f"Attempted synthetic: {attempted}")
    print(f"Added synthetic: {added}")
    print(f"Skipped synthetic: {skipped}")
    print(f"Final: {len(out_nl)}")
    print(f"Wrote NL: {args.out_nl}")
    print(f"Wrote SQL: {args.out_sql}")


if __name__ == "__main__":
    main()