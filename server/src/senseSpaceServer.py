# -----------------------------------------------------------------------------
# Sense Space
# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
# Zurich University of the Arts / zhdk.ch
# Max Rheiner, max.rheiner@zhdk.ch / 2025 
# -----------------------------------------------------------------------------


import socket
import threading
import json
import pyzed.sl as sl

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

def run_zed_fusion():
    # Initialize Fusion
    init_params = sl.InitFusionParameters()
    init_params.coordinate_units = sl.UNIT.METER

    fusion = sl.Fusion()
    status = fusion.init(init_params)
    if status != sl.FUSION_ERROR_CODE.SUCCESS:
        print("[ERROR] Fusion init failed:", status)
        return

    # Enable body tracking fusion
    bt_params = sl.BodyTrackingFusionParameters()
    bt_params.enable_tracking = True
    bt_params.enable_body_fitting = True
    fusion.enable_body_tracking(bt_params)

    print("[FUSION] Started multi-camera body tracking")

    runtime_params = sl.FusionRuntimeParameters()

    bodies = sl.Bodies()

    while True:
        # Process frames from all cameras
        fusion.process(runtime_params)

        # Retrieve fused bodies
        fusion.retrieve_bodies(bodies)

        people_data = []
        for person in bodies.body_list:
            skeleton = {}
            for i, kp in enumerate(person.keypoint):
                skeleton[f"joint_{i}"] = {
                    "x": float(kp[0]),
                    "y": float(kp[1]),
                    "z": float(kp[2]),
                }
            people_data.append({
                "id": person.id,
                "confidence": float(person.confidence),
                "skeleton": skeleton,
            })

        if people_data:
            broadcast({"people": people_data})

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    # Start TCP server in background
    threading.Thread(target=start_tcp_server, daemon=True).start()

    # Start ZED Fusion loop
    run_zed_fusion()
