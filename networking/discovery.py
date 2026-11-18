
import json
import socket
import threading
import time
import ipaddress

from networking.discovery_security import (
    create_announcement,
    verify_announcement
)

from networking.network_utils import (
    get_local_ipv4,
    get_broadcast_address
)


DISCOVERY_PORT = 37020
ANNOUNCEMENT_INTERVAL = 5

_stop_event = threading.Event()


def start_broadcaster(https_port=8000):
    """
    Broadcast authenticated PathSnap announcements.
    """

    local_ip = get_local_ipv4()

    if not local_ip:
        raise RuntimeError(
            "Unable to determine local network IPv4 address"
        )

    def broadcast_loop():
        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_BROADCAST,
            1
        )

        try:
            while not _stop_event.is_set():

                announcement = create_announcement(
                    local_ip,
                    https_port
                )

                message = json.dumps(
                    announcement
                ).encode("utf-8")

                try:
                    sock.sendto(
                        message,
                        (
                            get_broadcast_address(),
                            DISCOVERY_PORT
                        )
                    )

                except OSError:
                    pass

                _stop_event.wait(
                    ANNOUNCEMENT_INTERVAL
                )

        finally:
            sock.close()

    _stop_event.clear()

    thread = threading.Thread(
        target=broadcast_loop,
        daemon=True,
        name="PathSnapDiscovery"
    )

    thread.start()

    return thread


def stop_broadcaster():
    _stop_event.set()


def listen_for_servers(timeout=30):
    """
    Listen for authenticated PathSnap announcements.
    Intended for a separate Python discovery agent.
    """

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1
    )

    sock.bind(("", DISCOVERY_PORT))
    sock.settimeout(1)

    discovered = {}

    start_time = time.time()

    try:
        while time.time() - start_time < timeout:

            try:
                packet, sender = sock.recvfrom(4096)

                data = json.loads(packet.decode("utf-8"))

                if not verify_announcement(data):
                    continue

                advertised_ip = ipaddress.ip_address(
                    data["host"]
                )

                if not advertised_ip.is_private:
                    continue

                # Announcement address must match packet sender
                if data["host"] != sender[0]:
                    continue

                server_url = (
                    f"https://{data['host']}:{data['port']}"
                )

                discovered[server_url] = {
                    "host": data["host"],
                    "port": data["port"],
                    "url": server_url
                }

                print("PathSnap server discovered:")
                print(server_url)

            except socket.timeout:
                continue

            except (ValueError, UnicodeDecodeError):
                continue

    finally:
        sock.close()

    return list(discovered.values())


if __name__ == "__main__":
    listen_for_servers()
