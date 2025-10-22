"""
senseSpaceLib.senseSpace.communication

Shared TCP communication utilities for senseSpace clients and servers.
Supports MessagePack (binary) + Zstandard compression for optimal network performance.
"""

import socket
import threading
import json
import struct
from typing import Callable, Any, Optional

# Try to import msgpack, fall back to JSON only if not available
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    print("[WARNING] msgpack not installed. Install with: pip install msgpack")
    print("[WARNING] Falling back to JSON-only mode (slower, larger payload)")

# Try to import zstd for compression
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
    # Create reusable compressor/decompressor for better performance
    _zstd_compressor = zstd.ZstdCompressor(level=3)  # level 3 = fast, good compression
    _zstd_decompressor = zstd.ZstdDecompressor()
except ImportError:
    ZSTD_AVAILABLE = False
    print("[WARNING] zstandard not installed. Install with: pip install zstandard")
    print("[WARNING] Running without compression (larger bandwidth usage)")


# Protocol magic bytes to identify message format
MAGIC_MSGPACK = b'\x9f\xd0'           # 2-byte magic for MessagePack (uncompressed)
MAGIC_MSGPACK_ZSTD = b'\x9f\xd1'      # 2-byte magic for MessagePack + zstd
MAGIC_JSON = b'{'                      # JSON always starts with {


def serialize_message(data: dict, use_msgpack: bool = True, use_compression: bool = True) -> bytes:
    """
    Serialize a message to bytes using MessagePack + optional Zstandard compression.
    
    Args:
        data: Dictionary to serialize
        use_msgpack: If True and msgpack is available, use MessagePack. Otherwise use JSON.
        use_compression: If True and zstd is available, compress the payload.
    
    Returns:
        Serialized bytes with protocol header
    """
    if use_msgpack and MSGPACK_AVAILABLE:
        # MessagePack format: [magic:2 bytes][length:4 bytes][payload]
        payload = msgpack.packb(data, use_bin_type=True)
        
        # Apply compression if requested and available
        if use_compression and ZSTD_AVAILABLE:
            payload = _zstd_compressor.compress(payload)
            magic = MAGIC_MSGPACK_ZSTD
        else:
            magic = MAGIC_MSGPACK
        
        length = len(payload)
        return magic + struct.pack('>I', length) + payload
    else:
        # JSON format (legacy): just JSON with newline
        return (json.dumps(data) + "\n").encode("utf-8")


def deserialize_message(data: bytes) -> Optional[dict]:
    """
    Deserialize a message from bytes, auto-detecting format and compression.
    
    Args:
        data: Raw bytes received
    
    Returns:
        Deserialized dictionary or None on error
    """
    if len(data) == 0:
        return None
    
    # Check for MessagePack with compression
    if data[:2] == MAGIC_MSGPACK_ZSTD:
        if not MSGPACK_AVAILABLE or not ZSTD_AVAILABLE:
            print("[ERROR] Received compressed MessagePack but libraries not installed")
            return None
        try:
            if len(data) < 6:
                return None
            length = struct.unpack('>I', data[2:6])[0]
            compressed_payload = data[6:6+length]
            payload = _zstd_decompressor.decompress(compressed_payload)
            return msgpack.unpackb(payload, raw=False)
        except Exception as e:
            print(f"[ERROR] Compressed MessagePack deserialization failed: {e}")
            return None
    
    # Check for MessagePack without compression
    elif data[:2] == MAGIC_MSGPACK:
        if not MSGPACK_AVAILABLE:
            print("[ERROR] Received MessagePack data but msgpack is not installed")
            return None
        try:
            if len(data) < 6:
                return None
            length = struct.unpack('>I', data[2:6])[0]
            payload = data[6:6+length]
            return msgpack.unpackb(payload, raw=False)
        except Exception as e:
            print(f"[ERROR] MessagePack deserialization failed: {e}")
            return None
    else:
        # Assume JSON (legacy format)
        try:
            return json.loads(data.decode('utf-8').strip())
        except Exception as e:
            # Try to handle partial/multiple JSON objects separated by newlines
            try:
                lines = data.decode('utf-8').strip().split('\n')
                if lines:
                    return json.loads(lines[0])
            except Exception:
                pass
            return None


class TCPServer:
    def __init__(self, host: str, port: int, on_data: Callable[[dict, socket.socket], None], 
                 use_msgpack: bool = True, use_compression: bool = True):
        """
        TCP server for broadcasting data to multiple clients.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            on_data: Callback for received data: on_data(message_dict, client_socket)
            use_msgpack: Use MessagePack serialization (faster, smaller). Falls back to JSON if unavailable.
            use_compression: Use Zstandard compression (much smaller). Falls back to uncompressed if unavailable.
        """
        self.host = host
        self.port = port
        self.on_data = on_data
        self.clients = []
        self.running = False
        self.use_msgpack = use_msgpack and MSGPACK_AVAILABLE
        self.use_compression = use_compression and ZSTD_AVAILABLE
        
        if use_msgpack and not MSGPACK_AVAILABLE:
            print("[WARNING] MessagePack requested but not available, using JSON")
        if use_compression and not ZSTD_AVAILABLE:
            print("[WARNING] Compression requested but zstd not available")
        
        # Determine protocol description
        if self.use_msgpack and self.use_compression:
            self.protocol_name = "MessagePack+zstd"
        elif self.use_msgpack:
            self.protocol_name = "MessagePack"
        else:
            self.protocol_name = "JSON"

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen()
        print(f"[TCPServer] Listening on {self.host}:{self.port} (protocol: {self.protocol_name})")
        while self.running:
            try:
                conn, addr = s.accept()
                self.clients.append(conn)
                print(f"[TCPServer] Client connected: {addr}")
                threading.Thread(target=self._client_handler, args=(conn, addr), daemon=True).start()
            except Exception as e:
                if self.running:
                    print(f"[TCPServer] Accept error: {e}")

    def _client_handler(self, conn, addr):
        buffer = b''
        try:
            while self.running:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                
                buffer += chunk
                
                # Try to parse messages from buffer
                while buffer:
                    msg = None
                    consumed = 0
                    
                    # Check if it's MessagePack (compressed or uncompressed)
                    if buffer[:2] in (MAGIC_MSGPACK, MAGIC_MSGPACK_ZSTD):
                        if len(buffer) >= 6:
                            length = struct.unpack('>I', buffer[2:6])[0]
                            total_size = 6 + length
                            if len(buffer) >= total_size:
                                msg = deserialize_message(buffer[:total_size])
                                consumed = total_size
                    else:
                        # Assume JSON - look for newline
                        newline_idx = buffer.find(b'\n')
                        if newline_idx >= 0:
                            msg = deserialize_message(buffer[:newline_idx+1])
                            consumed = newline_idx + 1
                    
                    if msg:
                        try:
                            self.on_data(msg, conn)
                        except Exception as e:
                            print(f"[TCPServer] on_data callback error: {e}")
                    
                    if consumed > 0:
                        buffer = buffer[consumed:]
                    else:
                        break  # Wait for more data
                        
        except Exception as e:
            print(f"[TCPServer] Client handler error from {addr}: {e}")
        finally:
            if conn in self.clients:
                self.clients.remove(conn)
            conn.close()
            print(f"[TCPServer] Client disconnected: {addr}")

    def broadcast(self, data: dict):
        """Broadcast data to all connected clients"""
        msg = serialize_message(data, use_msgpack=self.use_msgpack, use_compression=self.use_compression)
        for c in list(self.clients):
            try:
                c.sendall(msg)
            except Exception as e:
                print(f"[TCPServer] Send error, removing client: {e}")
                if c in self.clients:
                    self.clients.remove(c)
                try:
                    c.close()
                except:
                    pass

    def stop(self):
        """Stop the server"""
        self.running = False


class TCPClient:
    def __init__(self, host: str, port: int, on_data: Callable[[dict], None], 
                 use_msgpack: bool = True, use_compression: bool = True):
        """
        TCP client for receiving data from server.
        
        Args:
            host: Server host
            port: Server port
            on_data: Callback for received data: on_data(message_dict)
            use_msgpack: Use MessagePack serialization (faster, smaller). Falls back to JSON if unavailable.
            use_compression: Use Zstandard compression (much smaller). Falls back to uncompressed if unavailable.
        """
        self.host = host
        self.port = port
        self.on_data = on_data
        self.sock = None
        self.running = False
        self.use_msgpack = use_msgpack and MSGPACK_AVAILABLE
        self.use_compression = use_compression and ZSTD_AVAILABLE
        
        if use_msgpack and not MSGPACK_AVAILABLE:
            print("[WARNING] MessagePack requested but not available, using JSON")
        if use_compression and not ZSTD_AVAILABLE:
            print("[WARNING] Compression requested but zstd not available")
        
        # Determine protocol description
        if self.use_msgpack and self.use_compression:
            self.protocol_name = "MessagePack+zstd"
        elif self.use_msgpack:
            self.protocol_name = "MessagePack"
        else:
            self.protocol_name = "JSON"

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.running = True
        print(f"[TCPClient] Connected to {self.host}:{self.port} (protocol: {self.protocol_name})")
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        buffer = b''
        try:
            while self.running:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                
                buffer += chunk
                
                # Try to parse messages from buffer
                while buffer:
                    msg = None
                    consumed = 0
                    
                    # Check if it's MessagePack (compressed or uncompressed)
                    if buffer[:2] in (MAGIC_MSGPACK, MAGIC_MSGPACK_ZSTD):
                        if len(buffer) >= 6:
                            length = struct.unpack('>I', buffer[2:6])[0]
                            total_size = 6 + length
                            if len(buffer) >= total_size:
                                msg = deserialize_message(buffer[:total_size])
                                consumed = total_size
                    else:
                        # Assume JSON - look for newline
                        newline_idx = buffer.find(b'\n')
                        if newline_idx >= 0:
                            msg = deserialize_message(buffer[:newline_idx+1])
                            consumed = newline_idx + 1
                    
                    if msg:
                        try:
                            self.on_data(msg)
                        except Exception as e:
                            print(f"[TCPClient] on_data callback error: {e}")
                    
                    if consumed > 0:
                        buffer = buffer[consumed:]
                    else:
                        break  # Wait for more data
                        
        except Exception as e:
            if self.running:
                print(f"[TCPClient] Listen error: {e}")
        finally:
            print(f"[TCPClient] Disconnected from {self.host}:{self.port}")

    def send(self, data: dict):
        """Send data to server"""
        msg = serialize_message(data, use_msgpack=self.use_msgpack, use_compression=self.use_compression)
        self.sock.sendall(msg)

    def close(self):
        self.running = False
        if self.sock:
            self.sock.close()


# ============================================================================
# UDP Broadcast (for low-latency streaming, packet loss OK)
# ============================================================================

class UDPBroadcaster:
    """
    UDP broadcaster for high-frequency, low-latency streaming.
    
    Perfect for real-time tracking data where:
    - Packet loss is acceptable (you have timestamps for ordering)
    - Low latency is critical
    - No connection overhead needed
    - Broadcasting to multiple clients
    
    Note: UDP has ~65KB packet size limit, so frames are automatically
    chunked if needed (though with MessagePack+zstd they should fit easily).
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 12346, 
                 use_msgpack: bool = True, use_compression: bool = True,
                 max_packet_size: int = 60000):
        """
        Args:
            host: Host to bind to (use "0.0.0.0" for all interfaces)
            port: Port to broadcast on
            use_msgpack: Use MessagePack serialization
            use_compression: Use Zstandard compression
            max_packet_size: Max UDP packet size (leave headroom for headers)
        """
        self.host = host
        self.port = port
        self.use_msgpack = use_msgpack and MSGPACK_AVAILABLE
        self.use_compression = use_compression and ZSTD_AVAILABLE
        self.max_packet_size = max_packet_size
        
        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
        
        # For multicast (alternative to broadcast)
        self.multicast_group = None
        
        if self.use_msgpack and self.use_compression:
            protocol_name = "UDP MessagePack+zstd"
        elif self.use_msgpack:
            protocol_name = "UDP MessagePack"
        else:
            protocol_name = "UDP JSON"
        
        print(f"[UDPBroadcaster] Ready on port {port} (protocol: {protocol_name})")
    
    def broadcast(self, data: dict, target_addresses=None):
        """
        Broadcast data via UDP.
        
        Args:
            data: Dictionary to broadcast
            target_addresses: Optional list of (ip, port) tuples. If None, uses broadcast.
        """
        # Serialize once
        msg_bytes = serialize_message(data, use_msgpack=self.use_msgpack, use_compression=self.use_compression)
        
        # Check if we need to chunk (rare with compression)
        if len(msg_bytes) > self.max_packet_size:
            print(f"[UDPBroadcaster] WARNING: Message too large ({len(msg_bytes)} bytes), chunking not implemented. Consider increasing compression or reducing data.")
            return
        
        # Broadcast to targets or use broadcast address
        if target_addresses:
            for addr in target_addresses:
                try:
                    self.sock.sendto(msg_bytes, addr)
                except Exception as e:
                    print(f"[UDPBroadcaster] Send error to {addr}: {e}")
        else:
            # Broadcast to subnet
            try:
                self.sock.sendto(msg_bytes, ('<broadcast>', self.port))
            except Exception as e:
                print(f"[UDPBroadcaster] Broadcast error: {e}")
    
    def close(self):
        """Close the UDP socket"""
        if self.sock:
            self.sock.close()


class UDPReceiver:
    """
    UDP receiver for high-frequency, low-latency streaming.
    
    Receives broadcast UDP packets and calls callback with deserialized data.
    Handles out-of-order packets via timestamps.
    """
    
    def __init__(self, on_data: Callable[[dict], None], port: int = 12346,
                 bind_address: str = "0.0.0.0"):
        """
        Args:
            port: Port to listen on
            on_data: Callback for received data: on_data(message_dict)
            bind_address: Address to bind to (use "0.0.0.0" for all interfaces)
        """
        self.port = port
        self.bind_address = bind_address
        self.on_data = on_data
        self.running = False
        self.sock = None
        
        print(f"[UDPReceiver] Initialized on port {port}")
    
    def start(self):
        """Start receiving UDP packets in a background thread"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_address, self.port))
        
        self.running = True
        threading.Thread(target=self._receive_loop, daemon=True).start()
        print(f"[UDPReceiver] Listening on {self.bind_address}:{self.port}")
    
    def _receive_loop(self):
        """Receive loop running in background thread"""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65536)  # Max UDP packet size
                if not data:
                    continue
                
                # Deserialize
                msg = deserialize_message(data)
                if msg:
                    try:
                        self.on_data(msg)
                    except Exception as e:
                        print(f"[UDPReceiver] Callback error: {e}")
            except Exception as e:
                if self.running:
                    print(f"[UDPReceiver] Receive error: {e}")
    
    def close(self):
        """Stop receiving and close socket"""
        self.running = False
        if self.sock:
            self.sock.close()
