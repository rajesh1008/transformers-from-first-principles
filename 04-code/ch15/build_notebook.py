"""Run lab15_cells.py cell by cell in one namespace, capture what each cell prints, and write
lab15.ipynb (nbformat 4, with the real outputs) and outputs/lab15.txt.  Standard library only.
Run from 04-code/ch15:  python3 build_notebook.py"""
import io, json, contextlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE)
src = open("lab15_cells.py").read().split("\n# %%")
cells, ns, log = [], {"__name__": "__lab15__"}, []
for k, chunk in enumerate(src[1:], 1):
    head, _, body = chunk.partition("\n")
    if head.strip() == "[markdown]":
        text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.strip().splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
        log.append(text)
        continue
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(body, f"lab15 cell {k}", "exec"), ns)
    out = buf.getvalue()
    cells.append({"cell_type": "code", "execution_count": len([c for c in cells if c["cell_type"] == "code"]) + 1,
                  "metadata": {}, "source": body.strip(),
                  "outputs": [{"output_type": "stream", "name": "stdout", "text": out}] if out else []})
    log.append(out.rstrip())
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("lab15.ipynb", "w"), indent=1)
open(os.path.join("outputs", "lab15.txt"), "w").write("\n".join(log) + "\n")
print("\n".join(log))
