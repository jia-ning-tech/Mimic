#!/usr/bin/env python3
import os, subprocess, sys, re, pathlib

root = pathlib.Path(__file__).resolve().parents[1]
readme = root / "README.md"
tree_txt = root / "project_tree.txt"
depth = int(os.environ.get("DEPTH", "3"))

def gen_tree():
    try:
        out = subprocess.check_output(["tree", "-L", str(depth), "--dirsfirst"], cwd=root, text=True)
    except Exception:
        print("[info] 'tree' not found, fallback to 'find'")
        out = subprocess.check_output(["find", ".", "-maxdepth", str(depth), "-print"], cwd=root, text=True)
        out = "\n".join(line.lstrip("./") for line in out.splitlines())
    tree_txt.write_text(out, encoding="utf-8")

def patch_readme():
    txt = readme.read_text(encoding="utf-8")
    block = "```text\n" + tree_txt.read_text(encoding="utf-8") + "\n```"
    if "<!-- BEGIN:PROJECT_TREE -->" in txt and "<!-- END:PROJECT_TREE -->" in txt:
        txt = re.sub(
            r"<!-- BEGIN:PROJECT_TREE -->.*?<!-- END:PROJECT_TREE -->",
            "<!-- BEGIN:PROJECT_TREE -->\n" + block + "\n<!-- END:PROJECT_TREE -->",
            txt,
            flags=re.S
        )
    else:
        txt += "\n\n## 📦 项目文件结构（自动生成）\n\n<!-- BEGIN:PROJECT_TREE -->\n" + block + "\n<!-- END:PROJECT_TREE -->\n"
    readme.write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    gen_tree()
    patch_readme()
    print("[ok] README updated with project tree")
