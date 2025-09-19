import os
import re
import graphviz
from pathlib import Path

# --- Config ---
ACTIVITY_DIR = Path("doc/llms/personas/activities")
OUTPUT_DIR = Path("doc/llms/graphs/activities")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Expected levels and order
LEVEL_ORDER = ["beginner", "intermediate", "advanced"]


def parse_activity_file(path: Path):
    """Parse a markdown activity file into headers -> list of activities."""
    text = path.read_text(encoding="utf-8")

    # Find all headers with their activities
    # Match "### Header" ... followed by "**Activities:**" block
    sections = []
    for header, block in re.findall(
        r"^###\s+(.*?)\n(?:\*\*Concept:\*\*.*\n)?\s*\*\*Activities:\*\*\n((?:-.*\n)+)"
        text,
        flags=re.MULTILINE | re.DOTALL,
    ):
        activities = [
            line.strip()[2:].strip()  # remove "- "
            for line in block.strip().splitlines()
            if line.strip().startswith("- ")
        ]
        sections.append((header.strip(), activities))

    return sections


def collect_by_category():
    """Return dict[category][level] = list of (header, activities)."""
    categories = {}

    for file in ACTIVITY_DIR.glob("*.md"):
        name = file.stem  # e.g. "coding_beginner"
        if "_" not in name:
            continue
        category, level = name.split("_", 1)
        sections = parse_activity_file(file)

        categories.setdefault(category, {})
        categories[category][level] = sections

    return categories


def build_graph(category, levels_dict):
    """Build a Graphviz graph for a single category."""
    dot = graphviz.Digraph(
        name=category,
        format="png",
        graph_attr={
            "rankdir": "LR",  # left-to-right
            "fontsize": "10",
        },
    )

    clusters = {}

    for i, level in enumerate(LEVEL_ORDER):
        if level not in levels_dict:
            continue
        cluster_name = f"cluster_{level}"
        with dot.subgraph(name=cluster_name) as sub:
            sub.attr(label=level.capitalize(), style="rounded", color="lightgrey")
            for header, activities in levels_dict[level]:
                node_id = f"{level}_{header.replace(' ', '_')}"
                label = f"{header}\n" + "\n".join(f"- {a}" for a in activities)
                sub.node(node_id, label=label, shape="box", style="rounded,filled", fillcolor="white")
        clusters[level] = cluster_name

    # Add edges beginner -> intermediate -> advanced
    for l1, l2 in zip(LEVEL_ORDER, LEVEL_ORDER[1:]):
        if l1 in clusters and l2 in clusters:
            dot.edge(f"{l1}_{levels_dict[l1][0][0].replace(' ', '_')}",
                     f"{l2}_{levels_dict[l2][0][0].replace(' ', '_')}",
                     style="invis")  # invisible edge just for alignment

    return dot


def main():
    categories = collect_by_category()
    for category, levels in categories.items():
        dot = build_graph(category, levels)
        out_path = OUTPUT_DIR / category
        dot.render(out_path, cleanup=True)
        print(f"Wrote {out_path}.png")


if __name__ == "__main__":
    main()

