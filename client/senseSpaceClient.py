import sys
import socket
import json
import threading
import argparse
from PyQt5.QtWidgets import QApplication
from libs.senseSpaceLib.senseSpace.protocol import Frame
from server.qt_viewer import MainWindow

def receive_thread(sock, on_frame):
    buffer = ""
    while True:
        data = sock.recv(4096)
        if not data:
            break
        buffer += data.decode('utf-8')
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            try:
                msg = json.loads(line)
                if msg.get("type") == "frame":
                    frame = Frame.from_dict(msg["data"])
                    on_frame(frame)
            except Exception as e:
                print(f"[ERROR] Failed to parse frame: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="Server IP or hostname")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"[CLIENT] Connected to {args.host}:{args.port}")

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    def on_frame(frame):
        # frame: Frame object
        people_data = [p.to_dict() for p in frame.people]
        win.update_people(people_data)

    threading.Thread(target=receive_thread, args=(sock, on_frame), daemon=True).start()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
