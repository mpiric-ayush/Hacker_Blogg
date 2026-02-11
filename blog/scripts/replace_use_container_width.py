import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def replace_in_text(path: Path, old: str, new: str) -> bool:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    if old not in txt:
        return False
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_text(txt, encoding="utf-8")
    txt = txt.replace(old, new)
    path.write_text(txt, encoding="utf-8")
    return True

def replace_in_ipynb(path: Path) -> bool:
    changed = False
    nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    for cell in nb.get("cells", []):
        if not isinstance(cell.get("source"), list):
            continue
        src = "".join(cell["source"])
        if "use_container_width=True" in src:
            src = src.replace("use_container_width=True", 'width="stretch"')
            changed = True
        if "use_container_width=False" in src:
            src = src.replace("use_container_width=False", 'width="content"')
            changed = True
        if changed:
            cell["source"] = [src]
    if changed:
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text(json.dumps(nb, indent=2), encoding="utf-8")
        path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    return changed

def main():
    changed_files = []
    for p in ROOT.rglob("*"):
        if p.suffix == ".ipynb":
            if replace_in_ipynb(p):
                changed_files.append(str(p))
        elif p.suffix in {".py", ".md", ".txt", ".env"}:
            updated = False
            updated |= replace_in_text(p, "use_container_width=True", 'width="stretch"')
            updated |= replace_in_text(p, "use_container_width=False", 'width="content"')
            if updated:
                changed_files.append(str(p))
    if changed_files:
        print("Updated files:")
        for f in changed_files:
            print(" -", f)
    else:
        print("No occurrences found. Repository already migrated.")

if __name__ == "__main__":
    main()
