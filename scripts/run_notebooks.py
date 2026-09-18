"""Execute the solution notebooks in place so students see real outputs."""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = f'{ROOT}/notebooks'

fails = 0
for f in sorted(os.listdir(NB)):
    if not f.endswith('_solution.ipynb'):
        continue
    path = os.path.join(NB, f)
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    try:
        NotebookClient(nb, timeout=600, kernel_name='python3',
                       resources={'metadata': {'path': NB}}).execute()
        nbformat.write(nb, path)
        n_out = sum(1 for c in nb.cells if c.cell_type == 'code' and c.get('outputs'))
        n_img = sum(1 for c in nb.cells if c.cell_type == 'code'
                    for o in c.get('outputs', [])
                    if 'image/png' in o.get('data', {}))
        print(f'  OK    {f:<48} {time.time()-t0:5.1f}s  {n_out} outputs, {n_img} figures')
    except Exception as e:
        fails += 1
        msg = str(e).strip().splitlines()
        print(f'  FAIL  {f}')
        for line in msg[-14:]:
            print('        ', line)
sys.exit(1 if fails else 0)
