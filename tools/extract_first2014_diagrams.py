"""Extract diagram geometry as an auditable draft, never as reviewed clinical rules.

Run with the bundled Python (pdfplumber). Text and vector coordinates come from
the user-provided First 2014 PDF. Unresolved edges remain explicit in the draft.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
import re


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def projection(point, segment):
    a, b = segment
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy) if dx or dy else 0
    t = max(0, min(1, t))
    return a[0] + t * dx, a[1] + t * dy


def box_distance(point, box):
    x, y = point
    left, top, right, bottom = box["bbox"]
    if left <= x <= right and top <= y <= bottom:
        return 0
    return math.hypot(max(left - x, 0, x - right), max(top - y, 0, y - bottom))


def rect_text(page, bbox):
    text = page.crop(bbox).extract_text(x_tolerance=1, y_tolerance=2) or ""
    return re.sub(r"\s+", " ", text).strip()


def extract_page(page):
    boxes = []
    for rect in page.rects:
        if not rect.get("stroke") or rect.get("linewidth", 0) < .5 or rect["width"] < 25 or rect["height"] < 10:
            continue
        bbox = [rect["x0"], rect["top"], rect["x1"], rect["bottom"]]
        if any(max(abs(a - b) for a, b in zip(bbox, old["bbox"])) < .5 for old in boxes):
            continue
        fill_rects = [r for r in page.rects if r.get("fill") and max(abs(a - b) for a, b in zip(
            bbox, [r["x0"], r["top"], r["x1"], r["bottom"]])) < .5]
        shaded = any(r.get("non_stroking_color") not in (None, 1, 1.0, (1, 1, 1)) for r in fill_rects)
        boxes.append({"bbox": bbox, "text": rect_text(page, bbox), "shaded": shaded})
    boxes.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    for index, box in enumerate(boxes):
        box["id"] = f"p{page.page_number:03d}_b{index:02d}"
    words = page.extract_words(x_tolerance=1, y_tolerance=2)
    outside = [w for w in words if w["top"] > 65 and not any(
        b["bbox"][0] - 1 <= (w["x0"] + w["x1"]) / 2 <= b["bbox"][2] + 1
        and b["bbox"][1] - 1 <= (w["top"] + w["bottom"]) / 2 <= b["bbox"][3] + 1 for b in boxes)]
    labels = [w for w in outside if w["text"] in {"Y", "N"}]
    free_text = [w for w in outside if w["text"] not in {"Y", "N"}]
    segments, arrows = [], []
    for line in page.lines:
        if line.get("stroke") and line.get("linewidth", 0) >= .4:
            segments.append(tuple(tuple(p) for p in line["pts"][:2]))
    for curve in page.curves:
        points = curve.get("pts", [])
        if curve.get("fill") and not curve.get("stroke") and curve["width"] <= 9 and curve["height"] <= 9 and len(points) >= 4:
            tip, tail = tuple(points[-2]), tuple(points[1])
            arrows.append({"tip": tip, "tail": tail})
        elif curve.get("stroke") and not curve.get("fill") and curve.get("linewidth", 0) >= .4:
            segments.extend((tuple(a), tuple(b)) for a, b in zip(points, points[1:]) if distance(a, b) > .1)

    # Build a connector graph from native vector endpoints and intersections.
    cuts = [[a, b] for a, b in segments]
    for i, (a, b) in enumerate(segments):
        for j in range(i):
            c, d = segments[j]
            candidates = [a, b, c, d]
            if abs(a[0] - b[0]) < .2 and abs(c[1] - d[1]) < .2:
                candidates.append((a[0], c[1]))
            if abs(a[1] - b[1]) < .2 and abs(c[0] - d[0]) < .2:
                candidates.append((c[0], a[1]))
            for point in candidates:
                if distance(point, projection(point, (a, b))) < 1 and distance(point, projection(point, (c, d))) < 1:
                    cuts[i].append(point)
                    cuts[j].append(point)
    for arrow in arrows:
        if segments:
            i = min(range(len(segments)), key=lambda k: distance(arrow["tail"], projection(arrow["tail"], segments[k])))
            point = projection(arrow["tail"], segments[i])
            arrow["connector_distance"] = distance(point, arrow["tail"])
            arrow["start"] = point
            cuts[i].append(point)
        if segments:
            j = min(range(len(segments)), key=lambda k: distance(arrow['tip'], projection(arrow['tip'], segments[k])))
            landing = projection(arrow['tip'], segments[j])
            if distance(landing, arrow['tip']) < 2.5:
                arrow['landing'] = landing
                cuts[j].append(landing)
    vertices, graph = [], {}

    def vertex(point):
        for i, other in enumerate(vertices):
            if distance(point, other) < 1.2:
                return i
        vertices.append(point)
        return len(vertices) - 1

    for points in cuts:
        origin = points[0]
        points = sorted(points, key=lambda p: distance(p, origin))
        for a, b in zip(points, points[1:]):
            ia, ib = vertex(a), vertex(b)
            if ia != ib:
                graph.setdefault(ia, {})[ib] = distance(a, b)
                graph.setdefault(ib, {})[ia] = distance(a, b)

    # Handbook flow charts move down the page. At a horizontal arrow into a
    # vertical trunk, the continuation is downwards, not back into its parent.
    forward, reverse = {}, {}
    for a, neighbors in graph.items():
        for b, length in neighbors.items():
            if vertices[b][1] >= vertices[a][1] - 1:
                forward.setdefault(a, {})[b] = length
                reverse.setdefault(b, {})[a] = length
    for arrow in arrows:
        if 'start' not in arrow:
            continue
        a = vertex(arrow['start'])
        b = vertex(arrow.get('landing', arrow['tip']))
        forward.setdefault(a, {})[b] = distance(vertices[a], vertices[b])
        reverse.setdefault(b, {})[a] = distance(vertices[a], vertices[b])

    def terminals(start, links, excluded):
        pending, costs, parents = [(0, start)], {start: 0}, {}
        found = []
        while pending:
            cost, current = heapq.heappop(pending)
            if cost != costs[current]:
                continue
            near = [b for b in boxes if b['id'] != excluded and box_distance(vertices[current], b) < 2]
            if near:
                box = min(near, key=lambda b: box_distance(vertices[current], b))
                back, v = [vertices[current]], current
                while v != start:
                    v = parents[v]
                    back.append(vertices[v])
                found.append((box['id'], list(reversed(back))))
                continue
            for neighbor, length in links.get(current, {}).items():
                trial = cost + length
                if trial < costs.get(neighbor, math.inf):
                    costs[neighbor], parents[neighbor] = trial, current
                    heapq.heappush(pending, (trial, neighbor))
        if not found:
            ends = [v for v in costs if len(links.get(v, {})) <= 1]
            if ends:
                v = max(ends, key=lambda v: costs[v])
                back = [vertices[v]]
                while v != start:
                    v = parents[v]
                    back.append(vertices[v])
                found.append((None, list(reversed(back))))
        return found

    edges = []
    for arrow in arrows:
        if "start" not in arrow:
            continue
        target_boxes = [b for b in boxes if box_distance(arrow["tip"], b) < 4]
        target = min(target_boxes, key=lambda b: box_distance(arrow["tip"], b))["id"] if target_boxes else None
        start = vertex(arrow["start"])
        origin_paths = terminals(start, reverse, target) or [(None, [arrow['start']])]
        dest_paths = ([(target, [arrow['tip']])] if target else
                      terminals(vertex(arrow.get('landing', arrow['tip'])), forward, None))
        target_text = None
        if target is None:
            lines = []
            for word in sorted(free_text, key=lambda w: (w["top"], w["x0"])):
                line = next((row for row in lines if abs(row[0]["top"] - word["top"]) < 2), None)
                if line is None:
                    lines.append([word])
                else:
                    line.append(word)
            groups = []
            for line in lines:
                current = []
                for word in sorted(line, key=lambda w: w["x0"]):
                    if current and word["x0"] - current[-1]["x1"] > 35:
                        groups.append(current)
                        current = []
                    current.append(word)
                if current:
                    groups.append(current)
            def text_gap(row):
                bbox = [min(w["x0"] for w in row), min(w["top"] for w in row),
                        max(w["x1"] for w in row), max(w["bottom"] for w in row)]
                return box_distance(arrow["tip"], {"bbox": bbox})
            if groups:
                first = min(groups, key=text_gap)
                if text_gap(first) < 14:
                    chosen = [first]
                    left = min(w["x0"] for w in first)
                    bottom = max(w["bottom"] for w in first)
                    for row in sorted(groups, key=lambda row: row[0]["top"]):
                        top = min(w["top"] for w in row)
                        if -1 < top - bottom < 7 and abs(min(w["x0"] for w in row) - left) < 12:
                            chosen.append(row)
                            bottom = max(w["bottom"] for w in row)
                    # Include an immediately preceding line if the arrow meets
                    # the second line of a multi-line unboxed destination.
                    for row in sorted(groups, key=lambda row: row[0]["top"], reverse=True):
                        top = min(w["top"] for w in chosen[0])
                        if -1 < top - max(w["bottom"] for w in row) < 7 and abs(min(w["x0"] for w in row) - left) < 12:
                            chosen.insert(0, row)
                    target_text = " ".join(w["text"] for row in chosen for w in sorted(row, key=lambda w: w["x0"]))
        for origin, origin_path in origin_paths:
            path_segments = list(zip(origin_path, origin_path[1:])) + [(arrow['tail'], arrow['tip'])]
            candidates = []
            for word in labels:
                point = ((word['x0'] + word['x1']) / 2, (word['top'] + word['bottom']) / 2)
                gap = min(distance(point, projection(point, segment)) for segment in path_segments)
                if gap < 13:
                    candidates.append((distance(point, origin_path[-1]) + gap, word['text']))
            candidates.sort()
            answer = candidates[0][1] if candidates else None
            for dest, dest_path in dest_paths or [(None, [arrow['tip']])]:
                edges.append({'source': origin, 'answer': answer, 'target': dest,
                              'target_text': target_text if not dest else None,
                              'tip': arrow['tip'], 'tail': arrow['tail'],
                              'connector_distance': arrow['connector_distance'],
                              'path': origin_path, 'destination_path': dest_path})
    return {"pdf_page": page.page_number, "print_page": page.page_number - 17,
            "boxes": boxes, "edges": edges,
            "free_words": [{k: w[k] for k in ("text", "x0", "x1", "top", "bottom")} for w in outside]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pages", default="40-172")
    args = parser.parse_args(argv)
    import pdfplumber
    selected = set()
    for part in args.pages.split(","):
        a, _, b = part.partition("-")
        selected.update(range(int(a), int(b or a) + 1))
    with pdfplumber.open(args.pdf) as pdf:
        pages = [extract_page(pdf.pages[i - 1]) for i in sorted(selected) if len(pdf.pages[i - 1].rects) > 1]
    result = {"review_status": "geometry_draft_not_executable", "source_pdf": str(Path(args.pdf).resolve()), "pages": pages}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Draft: {len(pages)} pages, {sum(len(p['boxes']) for p in pages)} boxes, {sum(len(p['edges']) for p in pages)} arrows")


if __name__ == "__main__":
    main()
