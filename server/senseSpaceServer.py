#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
# Zurich University of the Arts / zhdk.ch
# Max Rheiner, max.rheiner@zhdk.ch / 2025 
# -----------------------------------------------------------------------------


import socket, threading, json, time, argparse
import sys
import os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from senseSpace.protocol import Frame, Person, Joint
import pyzed.sl as sl
from OpenGL.GL import *
from OpenGL.GLU import *

# ---------------- TCP SERVER ---------------- #
HOST = "0.0.0.0"
PORT = 5000
clients = []

def client_handler(conn, addr):
    print(f"[CLIENT CONNECTED] {addr}")
    try:
        while True:
            data = conn.recv(1)  # keep alive
            if not data:
                break
    except:
        pass
    print(f"[CLIENT DISCONNECTED] {addr}")
    if conn in clients:
        clients.remove(conn)
    conn.close()

def broadcast(data):
    msg = (json.dumps(data) + "\n").encode("utf-8")
    for c in list(clients):
        try:
            c.sendall(msg)
        except:
            clients.remove(c)

def start_tcp_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    print(f"[SERVER] Listening on {HOST}:{PORT}")
    while True:
        conn, addr = s.accept()
        clients.append(conn)
        threading.Thread(target=client_handler, args=(conn, addr), daemon=True).start()

# ---------------- ZED FUSION ---------------- #
def run_fusion(viz_mode=False, update_callback=None):
    # Multi-camera ZED Fusion setup
    import os, glob
    init_params = sl.InitFusionParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # Look for latest calibration file in 'calib' folder
    calib_dir = os.path.join(os.path.dirname(__file__), "calib")
    latest_calib = None
    if os.path.isdir(calib_dir):
        calib_files = sorted(
            [f for f in glob.glob(os.path.join(calib_dir, "*.json"))],
            key=os.path.getmtime, reverse=True)
        if calib_files:
            latest_calib = calib_files[0]
            print(f"[FUSION] Using calibration file: {latest_calib}")

    # Discover and subscribe to all available cameras
    cam_list = sl.Camera.get_device_list()
    num_cams = len(cam_list)
    if num_cams == 0:
        print("[ERROR] No ZED cameras found. Please connect at least one camera and restart the server.")
        return
    fusion = sl.Fusion()
    if num_cams == 1:
        print("[FUSION] One camera found. No calibration file needed.")
        uuid = cam_list[0].serial_number
        print(f"[FUSION] Subscribing to camera {uuid}")
        fusion.subscribe(sl.CameraIdentifier(uuid), sl.CommunicationParameters(), sl.Transform())
    else:
        if latest_calib:
            # Try both possible attribute names for ZED SDK compatibility
            if hasattr(init_params, 'fusion_configuration_file_path'):
                init_params.fusion_configuration_file_path = latest_calib
            elif hasattr(init_params, 'calibration_file_path'):
                init_params.calibration_file_path = latest_calib
            else:
                print("[WARNING] No valid calibration file attribute found in InitFusionParameters.")
            print(f"[FUSION] {num_cams} cameras found. Subscribing all for fusion.")
            for cam in cam_list:
                uuid = cam.serial_number
                print(f"[FUSION] Subscribing to camera {uuid}")
                fusion.subscribe(sl.CameraIdentifier(uuid), sl.CommunicationParameters(), sl.Transform())
        else:
            print(f"[WARNING] {num_cams} cameras found but no calibration file in '{calib_dir}'. Using only the first camera.")
            uuid = cam_list[0].serial_number
            print(f"[FUSION] Subscribing to camera {uuid}")
            fusion.subscribe(sl.CameraIdentifier(uuid), sl.CommunicationParameters(), sl.Transform())
    status = fusion.init(init_params)
    if status != sl.FUSION_ERROR_CODE.SUCCESS:
        print("[ERROR] Fusion init failed:", status)
        return


    # Discover and subscribe to all available cameras
    cam_list = sl.Camera.get_device_list()
    if not cam_list:
        print("[ERROR] No ZED cameras found. Please connect at least one camera and restart the server.")
        return

    for cam in cam_list:
        uuid = cam.serial_number
        print(f"[FUSION] Subscribing to camera {uuid}")
        fusion.subscribe(sl.CameraIdentifier(uuid), sl.CommunicationParameters(), sl.Transform())

    bt_params = sl.BodyTrackingFusionParameters()
    bt_params.enable_tracking = True
    bt_params.enable_body_fitting = True
    # Set body format in a backward-compatible way (different SDK versions expose different names)
    try:
        if hasattr(bt_params, 'body_format'):
            bt_params.body_format = sl.BODY_FORMAT.BODY_34
        elif hasattr(bt_params, 'format'):
            bt_params.format = sl.BODY_FORMAT.BODY_34
        else:
            # Some SDK versions may not expose a direct attribute; ignore and rely on defaults
            pass
    except Exception as e:
        print(f"[WARNING] Could not set body format on BodyTrackingFusionParameters: {e}")
    fusion.enable_body_tracking(bt_params)

    print("[FUSION] Body tracking with BODY_34 started")

    # ZED SDK naming differs between versions; RuntimeParameters is commonly used for capture
    if hasattr(sl, 'FusionRuntimeParameters'):
        runtime_params = sl.FusionRuntimeParameters()
    else:
        runtime_params = sl.RuntimeParameters()
        # Parameters for body tracking retrieval (SDK requires BodyTrackingFusionRuntimeParameters)
        if hasattr(sl, 'BodyTrackingFusionRuntimeParameters'):
            bt_runtime_params = sl.BodyTrackingFusionRuntimeParameters()
        else:
            # fallback placeholder if not present
            bt_runtime_params = None
        bodies = sl.Bodies()

    MAX_FPS = 30
    FRAME_INTERVAL = 1.0 / MAX_FPS  # Max FPS for sending data to clients
    last_time = time.time()
    # subscriptions complete; ready to start fusion processing
    while True:
        now = time.time()
        if now - last_time < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - (now - last_time))
        last_time = time.time()

        # Fusion process (some SDK versions expect no args)
        fusion.process()
        # retrieve bodies requires a BodyTrackingFusionRuntimeParameters parameter
        if bt_runtime_params is not None:
            fusion.retrieve_bodies(bodies, bt_runtime_params)
        else:
            # older/newer SDKs may accept a single argument — try best-effort call
            try:
                fusion.retrieve_bodies(bodies)
            except TypeError:
                # give up for now and continue loop
                continue

        people = []
        for person in bodies.body_list:
            joints = [Joint(
                i=i,
                pos={"x": float(kp[0]), "y": float(kp[1]), "z": float(kp[2])},
                ori={"x": float(person.global_orientation[i][0]), "y": float(person.global_orientation[i][1]), "z": float(person.global_orientation[i][2]), "w": float(person.global_orientation[i][3])},
                conf=float(person.keypoint_confidence[i])
            ) for i, kp in enumerate(person.keypoint)]
            people.append(Person(
                id=person.id,
                tracking_state=str(person.tracking_state),
                confidence=float(person.confidence),
                skeleton=joints
            ))
        frame = Frame(
            timestamp=bodies.timestamp.get_seconds(),
            people=people,
            body_model="BODY_34"
        )

        # Throttle broadcast to max FPS
        broadcast({
            "type": "frame",
            "data": frame.to_dict()
        })


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["server", "viz"], default="server")
    args = parser.parse_args()

    threading.Thread(target=start_tcp_server, daemon=True).start()
    run_fusion(viz_mode=(args.mode == "viz"))
