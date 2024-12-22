import http.server
import socketserver
import subprocess
import json
import sys
import os

if len(sys.argv) == 2:
    bot = sys.argv[1]
    if not os.path.isfile(bot):
        print(f"error: {bot} does not exist.")
        sys.exit(1)
    if not os.access(bot, os.X_OK):
        print(f"error: {bot} is not executable.")
        sys.exit(1)
else:
    print(f"usage: {sys.argv[0]} [bot file]")
    sys.exit(1)

def get_computer_move(board):
    process = subprocess.Popen(
        [bot],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    input_str = " ".join(f"{n}" for n in board)
    print(f'{input_str = }')
    stdout, stderr = process.communicate(input=input_str)
    print(f"{stdout = }")
    if stderr:
        print(f"error running {bot}: {stderr}")
    try:
        lines = stdout.strip().split('\n')
        x, y = map(int, lines[0].split())
        debug = "\n".join(lines[1:])
        print(f"{x = }, {y = }, {debug = }")
        return {"x": x, "y": y, "debug": debug}
    except ValueError as e:
        print(f"error parsing output: {e}")
        return {"x": None, "y": None, "debug": str(e)}
    

class RequestHandler(http.server.SimpleHTTPRequestHandler):

    def _set_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type', 'application/json')

    def do_OPTIONS(self):
        self.send_response(200, "OK")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        print(f"received GET request for {self.path}")
        print("headers:")
        for header, value in self.headers.items():
            print(f"{header}: {value}")
        super().do_GET()

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        print(f"received POST request for {self.path}")
        print("headers:")
        for header, value in self.headers.items():
            print(f"{header}: {value}")
        print("body:")
        print(post_data)
        try:
            board = json.loads(post_data)
            print(f"parsed board: {board}")
        except json.JSONDecodeError as e:
            print(f"error parsing JSON: {e}")
            board = []
        move = get_computer_move(board)
        print(f"computer move: {move}")
        response = json.dumps(move)
        print(f"{response = }")
        self.wfile.write(response.encode('utf-8'))

PORT = 40000
with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
    print(f"serving at port {PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("shutting down server.")
        httpd.server_close()