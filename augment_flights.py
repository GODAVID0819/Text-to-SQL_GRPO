import argparse
import random
import re
from pathlib import Path
from typing import List, Tuple, Optional

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

NL_PREFIXES = [
    "show me", "list", "find", "give me", "what are", "which are"
]


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path: Path, lines: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def replace_all_case_insensitive(text: str, old: str, new: str) -> str:
    return re.sub(re.escape(old), new, text, flags=re.IGNORECASE)


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


def pick_distinct(pool: List[str], k: int, avoid: Optional[set] = None) -> List[str]:
    avoid = avoid or set()
    choices = [x for x in pool if x not in avoid]
    if len(choices) < k:
        choices = pool[:]
    return random.sample(choices, k) if k <= len(choices) else [random.choice(pool) for _ in range(k)]


def city_sql_to_nl(city: str) -> str:
    return city.lower()


def airline_sql_to_nl(code: str) -> str:
    return code.lower()


def format_date_nl(month: int, day: int) -> str:
    return f"{MONTHS[month - 1]} {day}"


def rebuild_nl(
    old_nl: str,
    from_cities: List[str],
    to_cities: List[str],
    stop_cities: List[str],
    airlines: List[str],
    day_names: List[str],
    date_info: Optional[Tuple[int, int, int]],
    arr_constraints: List[Tuple[str, int]],
    dep_constraints: List[Tuple[str, int]],
) -> str:
    prefix = random.choice(NL_PREFIXES)
    parts = [prefix]

    if from_cities and to_cities:
        parts.append(f"flights from {city_sql_to_nl(from_cities[0])} to {city_sql_to_nl(to_cities[0])}")
    elif from_cities:
        parts.append(f"flights from {city_sql_to_nl(from_cities[0])}")
    elif to_cities:
        parts.append(f"flights to {city_sql_to_nl(to_cities[0])}")
    else:
        parts.append("flights")

    if stop_cities:
        parts.append(f"with a stop in {city_sql_to_nl(stop_cities[0])}")

    if airlines:
        parts.append(f"on {airline_sql_to_nl(airlines[0])}")

    if date_info is not None:
        _, month, day = date_info
        parts.append(f"on {format_date_nl(month, day)}")
    elif day_names:
        parts.append(f"on {day_names[0].lower()}")

    for op, val in arr_constraints:
        parts.append(f"with arrival time {op} {val}")

    for op, val in dep_constraints:
        parts.append(f"with departure time {op} {val}")

    return normalize_ws(" ".join(parts))


def augment_pair(nl: str, sql: str) -> Tuple[str, str]:
    from_cities = extract_city_alias_values(sql, "city_1")
    to_cities = extract_city_alias_values(sql, "city_2")
    stop_cities = extract_city_alias_values(sql, "city_3")
    airlines = extract_airlines(sql)
    day_names = extract_day_names(sql)
    date_info = extract_date(sql)
    arr_constraints = extract_time_constraints(sql, "arrival_time")
    dep_constraints = extract_time_constraints(sql, "departure_time")

    new_sql = sql

    # cities
    used = set()
    if from_cities:
        new_from = pick_distinct(CITIES, len(from_cities), avoid=used)
        used.update(new_from)
        new_sql = replace_city_alias(new_sql, "city_1", new_from)
    else:
        new_from = from_cities[:]

    if to_cities:
        new_to = pick_distinct(CITIES, len(to_cities), avoid=used)
        used.update(new_to)
        new_sql = replace_city_alias(new_sql, "city_2", new_to)
    else:
        new_to = to_cities[:]

    if stop_cities:
        new_stop = pick_distinct(CITIES, len(stop_cities), avoid=used)
        used.update(new_stop)
        new_sql = replace_city_alias(new_sql, "city_3", new_stop)
    else:
        new_stop = stop_cities[:]

    # airlines
    if airlines:
        new_airlines = pick_distinct(AIRLINES, len(airlines))
        new_sql = replace_airlines(new_sql, new_airlines)
    else:
        new_airlines = airlines[:]

    # day/date
    if date_info is not None:
        year = date_info[0]
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        new_date = (year, month, day)
        new_sql = replace_date(new_sql, year, month, day)
        new_days = []
    else:
        new_date = None
        if day_names:
            new_days = pick_distinct(DAYS, len(day_names))
            new_sql = replace_day_names(new_sql, new_days)
        else:
            new_days = []

    # times
    if arr_constraints:
        new_arr = [random.choice([(op, val) for _, op, val in TIME_BUCKETS]) for _ in arr_constraints]
        new_sql = replace_time_constraints(new_sql, "arrival_time", new_arr)
    else:
        new_arr = []

    if dep_constraints:
        new_dep = [random.choice([(op, val) for _, op, val in TIME_BUCKETS]) for _ in dep_constraints]
        new_sql = replace_time_constraints(new_sql, "departure_time", new_dep)
    else:
        new_dep = []

    new_nl = rebuild_nl(
        old_nl=nl,
        from_cities=new_from,
        to_cities=new_to,
        stop_cities=new_stop,
        airlines=new_airlines,
        day_names=new_days,
        date_info=new_date,
        arr_constraints=new_arr,
        dep_constraints=new_dep,
    )

    return new_nl, normalize_ws(new_sql)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_nl", required=True)
    parser.add_argument("--train_sql", required=True)
    parser.add_argument("--out_nl", required=True)
    parser.add_argument("--out_sql", required=True)
    parser.add_argument("--copies_per_example", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    nl_lines = read_lines(Path(args.train_nl))
    sql_lines = read_lines(Path(args.train_sql))

    if len(nl_lines) != len(sql_lines):
        raise ValueError("Mismatched number of lines between NL and SQL files.")

    out_nl = list(nl_lines)
    out_sql = list(sql_lines)

    seen = set((normalize_ws(nl), normalize_ws(sql)) for nl, sql in zip(nl_lines, sql_lines))

    for nl, sql in zip(nl_lines, sql_lines):
        for _ in range(args.copies_per_example):
            aug_nl, aug_sql = augment_pair(nl, sql)
            key = (aug_nl, aug_sql)
            if key not in seen:
                seen.add(key)
                out_nl.append(aug_nl)
                out_sql.append(aug_sql)

    write_lines(Path(args.out_nl), out_nl)
    write_lines(Path(args.out_sql), out_sql)

    print(f"Original: {len(nl_lines)}")
    print(f"Final: {len(out_nl)}")
    print(f"Added: {len(out_nl) - len(nl_lines)}")


if __name__ == "__main__":
    main()