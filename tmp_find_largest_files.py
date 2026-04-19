import os
from pathlib import Path
root = Path('.')
files = []
for path in root.rglob('*'):
    if path.is_file():
        try:
            files.append((path.stat().st_size, str(path).replace('\\','/')))
        except OSError:
            pass
files.sort(reverse=True)
for size, path in files[:10]:
    print(size, path)
