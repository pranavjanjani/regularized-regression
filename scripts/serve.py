"""Static server for site/ that disables caching.

Plain `http.server` lets the browser hold on to stale CSS and JS, which makes
editing the course site confusing. This sends no-store on every response.
"""
import functools
import http.server
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'site')


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, fmt, *args):      # keep the console quiet
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    handler = functools.partial(Handler, directory=ROOT)
    print(f'serving {ROOT} on http://localhost:{port}  (caching disabled)')
    http.server.ThreadingHTTPServer(('127.0.0.1', port), handler).serve_forever()
