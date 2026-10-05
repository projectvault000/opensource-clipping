import json
from pathlib import Path


def test_quick_start_cells_are_parseable_and_rerun_safe():
    notebook = json.loads(Path("notebooks/Quick_Start.ipynb").read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, source in enumerate(code, 1):
        compile(source, f"Quick_Start cell {index}", "exec")

    combined = "\n".join(code)
    assert "rm -rf" not in combined
    assert "subprocess.run([sys.executable, 'main.py', *args], check=True)" in combined
    assert "--no-broll" in combined
    assert "manager.export_approved_set()" in combined
    assert "files.download(str(archive_path))" in combined
