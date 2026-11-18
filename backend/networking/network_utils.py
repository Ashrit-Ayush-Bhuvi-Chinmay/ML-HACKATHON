
import socket


def get_local_ipv4():
    """
    Determine the IPv4 address selected for outbound routing.
    UDP connect does not send application data.
    """

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    try:
        sock.connect(("192.0.2.1", 80))
        return sock.getsockname()[0]

    except OSError:
        return None

    finally:
        sock.close()


def get_broadcast_address():
    """
    Default limited broadcast address.
    """
    return "255.255.255.255"
