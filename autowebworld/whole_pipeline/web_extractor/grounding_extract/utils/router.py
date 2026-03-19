import re
from pathlib import Path
from typing import Dict


def load_routes(web_dir: Path) -> Dict[str, str]:
    """Load route mappings from router/index.js."""
    router_file = web_dir / "src/router/index.js"
    if not router_file.exists():
        return {}

    routes = {}
    content = router_file.read_text()

    pattern = r"\{\s*path:\s*['\"]([^'\"]+)['\"],\s*name:\s*['\"]([^'\"]+)['\"]"
    for match in re.finditer(pattern, content):
        path, name = match.groups()
        routes[name] = path

    return routes

