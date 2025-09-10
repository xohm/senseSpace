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
    win = MainWindow(server_ip=server.host, get_client_count=get_client_count)
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
    parser.add_argument("--host", default="0.0.0.0", help="TCP server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=12345, help="TCP server port (default: 12345)")
    
    args = parser.parse_args()
    
    # Create server instance
    server = SenseSpaceServer(host=args.host, port=args.port)
    
    try:
        # Initialize cameras
        print("[INFO] Initializing ZED cameras...")
        if not server.initialize_cameras():
            print("[ERROR] Failed to initialize cameras")
            return 1
        
        # Start TCP server
        print("[INFO] Starting TCP server...")
        server.start_tcp_server()
        
        if args.mode == "viz":
            # Run with Qt visualization
            exit_code = start_qt_application(server)
            return exit_code
        else:
            # Headless mode
            print("[INFO] Running in headless mode. Press Ctrl+C to exit...")
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
