#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
# Zurich University of the Arts / zhdk.ch
# Max Rheiner, max.rheiner@zhdk.ch / 2025 
# -----------------------------------------------------------------------------

import argparse
import threading
import time
from senseSpaceServer import start_tcp_server, run_fusion

# Optional import for Qt viewer
try:
    from qt_viewer import MainWindow, QApplication
    qt_available = True
except ImportError:
    qt_available = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["server", "viz"], default="server", help="Run in server (headless) or viz (Qt OpenGL) mode")
    args = parser.parse_args()

    # Start TCP server in background
    from senseSpaceServer import HOST, clients
    threading.Thread(target=start_tcp_server, daemon=True).start()

    if args.mode == "viz":
        if not qt_available:
            print("[ERROR] Qt viewer not available. Install pyqt5 and try again.")
            return
        # Start Qt app and pass a callback to update the viewer
        import sys
        from qt_viewer import MainWindow
        from PyQt5.QtWidgets import QApplication
        def get_client_count():
            return len(clients)
        app = QApplication(sys.argv)
        win = MainWindow(server_ip=HOST, get_client_count=get_client_count)
        win.show()
        # Start fusion in a thread, update Qt viewer
        def fusion_thread():
            def update_callback(people_data):
                win.update_people(people_data)
            run_fusion(viz_mode=False, update_callback=update_callback)
        threading.Thread(target=fusion_thread, daemon=True).start()
        sys.exit(app.exec_())
    else:
        # Headless mode
        run_fusion(viz_mode=False)

if __name__ == "__main__":
    main()
