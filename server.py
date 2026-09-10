import http.server
import socketserver
import webbrowser
import os

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/dashboard.html"
        print("=" * 70)
        print("DYNAFLOW™ URBAN MOBILITY DASHBOARD ATTIVA")
        print(f"👉 Apri il browser all'indirizzo: {url}")
        print("Premi CTRL+C nel terminale per arrestare il server")
        print("=" * 70)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer arrestato.")

if __name__ == '__main__':
    start_server()
