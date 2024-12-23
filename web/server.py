import http.server
import socketserver
import subprocess
import json
import sys
import os

if len(sys.argv) == 2:
    bot = sys.argv[1]
    if not os.path.isfile(bot):
        print(f"Error: {bot} does not exist.")
        sys.exit(1)
    if not os.access(bot, os.X_OK):
        print(f"Error: {bot} is not executable.")
        sys.exit(1)
else:
    print(f"Usage: {sys.argv[0]} [bot file]")
    sys.exit(1)

def get_computer_move(board):
    process = subprocess.Popen(
        [bot],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    input_str = f"{board[0]}\n" + "\n".join(f"{board[i]} {board[i+1]}" for i in range(1, len(board), 2)) + "\n"
    print(f'{input_str = }')
    stdout, stderr = process.communicate(input=input_str)
    print(f"{stdout = }")
    if stderr:
        print(f"Error running {bot}: {stderr}")
    try:
        lines = stdout.strip().split('\n')
        x, y = map(int, lines[0].split())
        if len(lines) >= 2:
            debug = lines[1]
        else:
            debug = ""
        print(f"{x = }, {y = }, {debug = }")
        return {"x": x, "y": y, "debug": debug}
    except ValueError as e:
        print(f"Error parsing output: {e}")
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
        print(f"Received GET request for {self.path}")
        print("Headers:")
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
        print(f"Received POST request for {self.path}")
        print("Headers:")
        for header, value in self.headers.items():
            print(f"{header}: {value}")
        print("Body:")
        print(post_data)
        try:
            board = json.loads(post_data)
            print(f"Parsed board: {board}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            board = []
        move = get_computer_move(board)
        print(f"Computer move: {move}")
        response = json.dumps(move)
        print(f"{response = }")
        self.wfile.write(response.encode('utf-8'))

with open('config.json') as config_file:
    config = json.load(config_file)
    port = config['port']
    print(f'Using port: {port}')

PORT = port
with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
    print(f"Serving...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server.")
        httpd.server_close()
