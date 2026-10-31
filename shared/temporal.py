"""可复核的中英文病程区间解析和三值比较。区间宽度是版本化工程约定。"""
from __future__ import annotations

from datetime import date
from calendar import monthrange
import math
import re

from shared.clinical_contract import DurationEvidence, TemporalDecision

DAY_FACTORS = {"day": 1, "week": 7, "month": 30.44, "year": 365.25}
_UNITS = {"天": "day", "日": "day", "周": "week", "星期": "week", "礼拜": "week",
          "月": "month", "年": "year", "day": "day", "week": "week", "month": "month", "year": "year"}
_NUM = r"(?:\d+(?:\.\d+)?|[零〇一二两俩三四五六七八九十百千万]+|half|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|半)"
_DURATION = re.compile(rf"(?P<num>{_NUM})(?:\s*(?:个|個))?(?P<half>半|\s+and\s+a\s+half)?(?P<before_more>多)?(?:\s+a)?\s*(?P<unit>星期|礼拜|周|天|日|月|年|days?|weeks?|months?|years?)(?P<tail>半|多)?", re.I)
_WORDS = {w: i for i, w in enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"))}
_DIGITS = dict(zip("零〇一二两俩三四五六七八九", [0, 0, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9]))


def number(text: str) -> float:
    if text in ("半", "half"):
        return .5
    if text.lower() in _WORDS:
        return float(_WORDS[text.lower()])
    try:
        return float(text)
    except ValueError:
        total, current = 0, 0
        for char in text:
            if char in _DIGITS:
                current = _DIGITS[char]
            elif char in "十百千万":
                magnitude = {"十": 10, "百": 100, "千": 1000, "万": 10000}[char]
                total += (current or 1) * magnitude
                current = 0
            else:
                raise ValueError("无法识别数词")
        return float(total + current)


def to_days(quantity: float, unit: str) -> int:
    """按日精度保存；月向下取整，年四舍五入。门槛使用同一精度。"""
    value = quantity * DAY_FACTORS[unit]
    round_year = unit == "year" and float(quantity).is_integer()
    return math.floor(value + .5) if round_year else math.floor(value + 1e-9)


def parse_duration(text: str, *, symptom_id="", scope="unknown", evidence_refs=None, source="L2",
                   interview_date: date | str | None = None, event_dates: dict[str, str] | None = None,
                   pattern: str | None = None) -> DurationEvidence:
    interview = date.fromisoformat(interview_date) if isinstance(interview_date, str) else interview_date
    value = text.strip()
    lower = value.lower()
    observed_pattern = ("episodic" if re.search(r"反复|时好时坏|每次|间歇|episodes?|on and off|comes and goes", lower)
                        else "continuous" if re.search(r"持续|一直|连续|continuously|continuous|every day", lower) else "unknown")
    record = DurationEvidence(raw_span=value or "（未提供病程原话）", symptom_id=symptom_id, scope=scope,
        evidence_refs=evidence_refs or [], source=source, interview_date=interview,
        pattern=pattern or observed_pattern,
        duration_kind="episode" if re.search(r"每次|each episode|per episode", lower) else "overall")
    calendar_start = re.search(r"(?P<year>20\d{2})(?:年|[-/])(?P<month>\d{1,2})(?:月|[-/])(?:(?P<day>\d{1,2})日?)?", lower)
    relative_month = re.search(r"(去年|今年|前年)(\d{1,2})月", lower)
    year_start = re.search(r"(20\d{2})年(?:开始|起)", lower)
    month_only = re.search(r"(?:从)?\d{1,2}月(?:\d{1,2}日)?(?:开始|起)", lower)
    if calendar_start or relative_month or month_only or year_start:
        record.unit, record.anchor, record.parse_status = "relative", "calendar_start", "unresolved"
        if interview and (calendar_start or relative_month or year_start):
            try:
                year = int(calendar_start["year"]) if calendar_start else int(year_start[1]) if year_start else interview.year - {"今年": 0, "去年": 1, "前年": 2}[relative_month[1]]
                month = int(calendar_start["month"]) if calendar_start else 1 if year_start else int(relative_month[2])
                day = int(calendar_start["day"]) if calendar_start and calendar_start["day"] else None
                start = date(year, month, day or 1)
                end = date(year, 12, 31) if year_start and not calendar_start else date(year, month, day or monthrange(year, month)[1])
                if start <= interview:
                    record.min_days = max(0, (interview - end).days)
                    record.max_days = (interview - start).days
                    record.parse_status, record.approx = "ok", day is None
            except ValueError:
                record.parse_status = "failed"
        return record
    if re.search(r"去年|前年|今年|冬天|春天|夏天|秋天|换工作|高三|since\s+(?:last|my|the)", lower) and not _DURATION.search(lower):
        record.unit, record.anchor, record.parse_status = "relative", "event_date_required", "unresolved"
        if "去年冬天" in value and interview:
            start, end = date(interview.year - 1, 12, 1), date(interview.year, 3, 1)
            if end <= interview:
                record.min_days = (interview - end).days + 1
                record.max_days = (interview - start).days
                record.parse_status, record.approx, record.anchor = "ok", True, "interview:last_winter_dec_feb"
        for event, when in (event_dates or {}).items():
            if event in value and interview:
                # “以后”只给起点上界，不能当作事件当天开始。
                elapsed = (interview - date.fromisoformat(when)).days
                if elapsed >= 0:
                    record.min_days, record.max_days = 0, elapsed
                    record.parse_status, record.approx, record.anchor = "ok", True, f"event:{event}"
        return record
    vague_quantity = re.search(r"几[个]?(?:月|周|天|年)|数月|(?:a few|several|a couple(?: of)?)\s+(?:days?|weeks?|months?|years?)", lower)
    weak_time = re.search(r"好一阵|一阵子|很久|好久|a (?:little )?while|long time|rough semester|lately", lower)
    if vague_quantity or (weak_time and not _DURATION.search(lower)):
        record.parse_status, record.approx = "vague", True
        return record
    if "大半年" in value:
        record.min_days, record.max_days, record.quantity, record.unit = 150, 240, .5, "year"
        record.parse_status, record.approx = "ok", True
        record.approximation = "colloquial"
        return record
    interval = re.search(rf"({_NUM})\s*(?:到|至|[-~～—–])\s*({_NUM})(?:个)?\s*(星期|周|天|月|年|days?|weeks?|months?|years?)", lower)
    if interval:
        unit = _UNITS[interval[3].rstrip("s")]
        record.min_days, record.max_days = to_days(number(interval[1]), unit), to_days(number(interval[2]), unit)
        if record.min_days > record.max_days:
            record.min_days = record.max_days = None
            record.parse_status = "failed"
        else:
            record.unit, record.parse_status, record.approx, record.approximation = unit, "ok", True, "colloquial"
        return record
    matches = list(_DURATION.finditer(lower))
    if not matches:
        return record
    # 不把发作次数、年龄或睡眠小时数当作总病程；每个跨度单独输出。
    values = []
    for match in matches:
        quantity = number(match["num"])
        if match["half"] or match["tail"] == "半":
            quantity += .5
        unit = _UNITS[match["unit"].rstrip("s")]
        values.append((quantity, unit, match))
    if len(values) > 1:
        # 只有“1年两个月”这类紧邻复合时长可相加，其余需定向拆分。
        if any(lower[a[2].end():b[2].start()].strip("零又 ") for a, b in zip(values, values[1:])):
            record.parse_status = "vague"
            return record
        record.min_days = record.max_days = math.floor(sum(n * DAY_FACTORS[u] for n, u, _ in values) + 1e-9)
        record.unit, record.parse_status = "day", "ok"
        return record
    quantity, unit, match = values[0]
    if len(match["num"]) == 2 and all(c in _DIGITS for c in match["num"]):
        a, b = (_DIGITS[c] for c in match["num"])
        if a <= b:
            record.min_days, record.max_days = to_days(a, unit), to_days(b, unit)
            record.unit, record.parse_status, record.approx, record.approximation = unit, "ok", True, "colloquial"
        return record
    central = quantity * DAY_FACTORS[unit]
    exact = to_days(quantity, unit)
    record.quantity, record.unit, record.min_days, record.max_days = quantity, unit, exact, exact
    record.parse_status = "ok"
    more = match["tail"] == "多" or bool(match["before_more"]) or bool(re.search(r"more than|over\s+" + re.escape(match.group()), lower))
    under = bool(re.search(r"不到|不足|less than|under", lower))
    near = bool(re.search(r"将近|接近|nearly|almost", lower))
    around = bool(re.search(r"约|差不多|左右|大概|大约|about|around|approximately|roughly|maybe|probably|or so", lower))
    if more:
        record.min_days = exact
        record.max_days = 540 if unit == "year" and quantity == 1 else to_days(quantity + (1 if unit != "year" else .5), unit)
    elif under or near:
        record.min_days = math.floor(central * .9) if near else 0
        record.max_days = max(0, exact - 1)
    elif around:
        if unit == "week":
            record.min_days, record.max_days = max(0, exact - 4), exact + 4
        else:
            record.min_days, record.max_days = math.ceil(central * .9), math.floor(central * 1.1)
    record.approx = more or under or near or around
    record.approximation = "more" if more else "under" if under else "near" if near else "around" if around else "none"
    return record


def compare_duration(record: DurationEvidence, threshold: str | float, *, operator="min",
                     required_pattern="unknown", required_kind="overall") -> TemporalDecision:
    target = parse_duration(threshold, source="legacy_adapter") if isinstance(threshold, str) else None
    days = target.min_days if target and target.parse_status == "ok" and not target.approx else threshold if isinstance(threshold, (int, float)) else None
    def result(status, reason):
        return TemporalDecision(status=status, threshold_days=days, reason=reason, evidence_refs=record.evidence_refs)
    if days is None or record.parse_status != "ok":
        return result("unknown", "病程或门槛尚未解析为可比较区间")
    if required_kind != "unknown" and record.duration_kind != required_kind:
        return result("unknown", "单次发作时长与整体病程不匹配")
    if required_pattern != "unknown" and record.pattern != required_pattern:
        return result("unknown", "病程模式尚未证实符合该门要求")
    lo, hi = record.min_days, record.max_days
    if operator == "min":
        if lo is not None and lo >= days:
            return result("met", "区间下界达到门槛")
        if hi is not None and hi < days:
            return result("not_met", "区间上界仍低于门槛")
    elif operator == "max":
        if hi is not None and hi <= days:
            return result("met", "区间上界不超过门槛")
        if lo is not None and lo > days:
            return result("not_met", "区间下界超过允许上限")
    if record.approximation == "around" and record.quantity is not None and record.unit in DAY_FACTORS and to_days(record.quantity, record.unit) == days:
        return result("met_approx", "约数中心落在门槛上；策略v1要求澄清，暂不作为已满足")
    return result("unknown", "区间跨过门槛或缺少一侧边界")


def parse_duration_records(text: str, **kwargs) -> list[DurationEvidence]:
    """整体反复病程和每次发作时长分别保存，不相加。"""
    episode = re.search(r"每次|each episode|per episode", text, re.I)
    if episode:
        before = text[:episode.start()].strip(" ，,。;；")
        after = text[episode.start():].strip()
        if _DURATION.search(before) and _DURATION.search(after):
            return [parse_duration(before, **kwargs), parse_duration(after, **kwargs)]
    return [parse_duration(text, **kwargs)]
