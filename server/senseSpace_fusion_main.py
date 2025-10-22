#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Server Qt Application Launcher
# -----------------------------------------------------------------------------
# Zurich University of the Arts / zhdk.ch
# Max Rheiner, max.rheiner@zhdk.ch / 2025 
# -----------------------------------------------------------------------------

import argparse
import sys
import threading
import time

# Import from our shared library
import os, sys
# Ensure local 'libs' folder is on sys.path when running from repo
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
libs_path = os.path.join(repo_root, 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

from senseSpaceLib.senseSpace.server import SenseSpaceServer

# Optional import for Qt viewer
try:
    from qt_viewer import MainWindow
    from PyQt5.QtWidgets import QApplication
    qt_available = True
except ImportError:
    qt_available = False


def start_qt_application(server: SenseSpaceServer):
    """Start Qt application with the given server"""
    if not qt_available:
        print("[ERROR] Qt viewer not available. Install PyQt5 and try again.")
        return 1
    
    app = QApplication(sys.argv)
    
    def get_client_count():
        return len(server.clients)
    
    # Create main window
    win = MainWindow(server=server, server_ip=server.host, server_port=server.port, get_client_count=get_client_count)
    win.show()
    
    # Set up server callback to update Qt viewer
    def qt_update_callback(people_data, floor_height):
        """Callback to update Qt viewer from server thread"""
        # Try to emit signals; if the window or widget has been deleted, sip/PyQt will raise
        # a RuntimeError (wrapped C/C++ object has been deleted). In that case we should
        # silently ignore updates because the UI is gone.
        try:
            win.people_signal.emit(people_data)
            if hasattr(win, 'floor_height_signal') and floor_height is not None:
                win.floor_height_signal.emit(floor_height)
            elif hasattr(win, 'set_floor_height') and floor_height is not None:
                win.set_floor_height(floor_height)
        except RuntimeError:
            # Window/widget deleted: ignore update
            return
        except Exception:
            # Other exceptions: try a safe direct call, but guard against deleted objects
            try:
                win.update_people(people_data)
            except RuntimeError:
                return
            except Exception:
                # give up on this update
                return
    
    # Set the callback and start body tracking in background thread
    server.set_update_callback(qt_update_callback)
    tracking_thread = threading.Thread(target=server.run_body_tracking_loop, daemon=True)
    tracking_thread.start()
    
    # Run Qt event loop
    return app.exec_()


def main():
    """Legacy main function for backward compatibility"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["server", "viz"], default="server", 
                       help="Run in server (headless) or viz (Qt OpenGL) mode")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=12345, help="Server port (default: 12345)")
    parser.add_argument("--tcp", action="store_true", help="Use TCP instead of UDP for broadcasting (default: UDP)")
    parser.add_argument("--no-cameras", action="store_true", help="Skip ZED camera initialization and start server only (debug)")
    parser.add_argument("--viz", action="store_true", help="Enable visualization (alternative to --mode viz)")
    
    args = parser.parse_args()
    
    # Handle --viz flag as alternative to --mode viz
    if args.viz:
        args.mode = "viz"
    
    # Create server instance (UDP by default, TCP if --tcp flag is set)
    server = SenseSpaceServer(host=args.host, port=args.port, use_udp=not args.tcp)
    
    try:
        if args.mode == "viz":
            # Visualization mode: initialize cameras and run Qt viewer
            if not args.no_cameras:
                print("[INFO] Initializing ZED cameras...")
                if not server.initialize_cameras():
                    print("[ERROR] Failed to initialize cameras")
                    return 1
            else:
                print("[INFO] Skipping camera initialization (debug mode)")
            
            # Start TCP server
            print("[INFO] Starting TCP server...")
            server.start_tcp_server()
            
            # Run with Qt visualization
            exit_code = start_qt_application(server)
            return exit_code
        else:
            # Headless server mode
            print("[INFO] Running in headless mode. Press Ctrl+C to exit...")
            
            # Initialize cameras only if not in debug mode
            if not args.no_cameras:
                print("[INFO] Initializing ZED cameras...")
                if not server.initialize_cameras():
                    print("[ERROR] Failed to initialize cameras")
                    return 1
            else:
                print("[INFO] Skipping camera initialization (debug mode)")
            
            # Start TCP server
            print("[INFO] Starting TCP server...")
            server.start_tcp_server()
            
            # If cameras were not initialized (debug --no-cameras), run a simple loop
            # to keep the TCP server alive so clients can connect for debugging.
            if args.no_cameras:
                 try:
                     while server.running:
                         # Sleep briefly and allow server threads to operate
                         time.sleep(0.5)
                     return 0
                 except KeyboardInterrupt:
                     return 0
            else:
                # Run body tracking loop in headless mode
                 server.run_body_tracking_loop()
                 return 0
            
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
        return 1
    finally:
        server.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
