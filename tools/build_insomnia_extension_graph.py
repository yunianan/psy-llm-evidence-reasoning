"""Export the standalone extension subgraph without changing its source JSON."""
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.insomnia_subgraph import InsomniaExtensionSubgraph
from shared.knowledge_sources import SUPPLEMENT_ROOT


if __name__ == "__main__":
    graph = InsomniaExtensionSubgraph().to_dict()
    target = SUPPLEMENT_ROOT.parent / "extension_graph.json"
    target.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{target}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
