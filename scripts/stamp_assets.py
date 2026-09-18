"""Stamp site asset URLs with a content hash, e.g. styles.css?v=1a2b3c4d.

GitHub Pages serves assets with Cache-Control: max-age=600, so for ten minutes
after a deploy a returning visitor can run the previous CSS or JS against the new
HTML. Keying the URL to the file's contents makes every change cache-bust itself
immediately, and an unchanged file keeps its old URL so it stays cached.

Idempotent: existing ?v= stamps are stripped and recomputed. Run from
scripts/sync_site.sh, after the site folder is assembled.
"""
import hashlib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, 'site')
REF = re.compile(r'(?P<attr>href|src)="(?P<path>assets/[^"?#]+\.(?:css|js))(?:\?v=[0-9a-f]+)?"')


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:8]


def main():
    cache, changed = {}, 0
    for name in sorted(os.listdir(SITE)):
        if not name.endswith('.html'):
            continue
        page = os.path.join(SITE, name)
        src = open(page).read()

        def sub(m):
            rel = m.group('path')
            full = os.path.join(SITE, rel)
            if not os.path.exists(full):
                return m.group(0)
            if rel not in cache:
                cache[rel] = digest(full)
            return f'{m.group("attr")}="{rel}?v={cache[rel]}"'

        out = REF.sub(sub, src)
        if out != src:
            open(page, 'w').write(out)
            changed += 1
    print(f'stamped {len(cache)} assets across {changed} pages')


if __name__ == '__main__':
    main()
