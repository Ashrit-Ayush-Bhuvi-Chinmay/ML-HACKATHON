
import argparse
import ipaddress
import getpass
from pathlib import Path
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa


BASE_DIR = Path(__file__).resolve().parents[1]
CERT_DIR = BASE_DIR / "certificates"


def generate_certificates(extra_ip=None):

    CERT_DIR.mkdir(parents=True, exist_ok=True)

    ca_key_path = CERT_DIR / "ca.key.pem"
    ca_cert_path = CERT_DIR / "ca.crt.pem"
    server_key_path = CERT_DIR / "server.key.pem"
    server_cert_path = CERT_DIR / "server.crt.pem"

    existing = [
        ca_key_path, ca_cert_path,
        server_key_path, server_cert_path
    ]

    if any(path.exists() for path in existing):
        raise FileExistsError(
            "Certificates already exist. Back them up before regenerating."
        )

    # Generate private CA key
    ca_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096
    )

    # Generate server key
    server_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=3072
    )

    ca_name = x509.Name([
        x509.NameAttribute(
            NameOID.COUNTRY_NAME, "IN"
        ),
        x509.NameAttribute(
            NameOID.ORGANIZATION_NAME, "PathSnap Team"
        ),
        x509.NameAttribute(
            NameOID.COMMON_NAME, "PathSnap Private Root CA"
        )
    ])

    server_name = x509.Name([
        x509.NameAttribute(
            NameOID.ORGANIZATION_NAME, "PathSnap"
        ),
        x509.NameAttribute(
            NameOID.COMMON_NAME, "pathsnap.local"
        )
    ])

    now = datetime.now(timezone.utc)

    # Create root CA certificate
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(
            x509.BasicConstraints(
                ca=True,
                path_length=0
            ),
            critical=True
        )
        .add_extension(
    x509.SubjectKeyIdentifier.from_public_key(
        ca_key.public_key()
    ),
    critical=False
)
.add_extension(
    x509.AuthorityKeyIdentifier.from_issuer_public_key(
        ca_key.public_key()
    ),
    critical=False
)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True
        )
        .sign(ca_key, hashes.SHA256())
    )

    # Server certificate names
    san_entries = [
        x509.DNSName("localhost"),
        x509.DNSName("pathsnap.local"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1"))
    ]

    if extra_ip:
        san_entries.append(
            x509.IPAddress(ipaddress.ip_address(extra_ip))
        )

    # Create server certificate
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(san_entries),
            critical=False
        )
        .add_extension(
            x509.BasicConstraints(
                ca=False,
                path_length=None
            ),
            critical=True
        )
        .add_extension(
    x509.SubjectKeyIdentifier.from_public_key(
        server_key.public_key()
    ),
    critical=False
)
.add_extension(
    x509.AuthorityKeyIdentifier.from_issuer_public_key(
        ca_key.public_key()
    ),
    critical=False
)
        .add_extension(
            x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.SERVER_AUTH
            ]),
            critical=False
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True
        )
        .sign(ca_key, hashes.SHA256())
    )

    # Encrypt CA private key with a password
    password = getpass.getpass(
        "Create a password for the private CA key: "
    )

    if len(password) < 12:
        raise ValueError("Use a password of at least 12 characters.")

    confirm = getpass.getpass("Confirm CA password: ")

    if password != confirm:
        raise ValueError("Passwords do not match.")

    # Save encrypted CA private key
    ca_key_path.write_bytes(
        ca_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(
                password.encode()
            )
        )
    )

    # Save public CA certificate
    ca_cert_path.write_bytes(
        ca_cert.public_bytes(serialization.Encoding.PEM)
    )

    # Save server private key
    server_key_path.write_bytes(
        server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
    )

    # Save server certificate
    server_cert_path.write_bytes(
        server_cert.public_bytes(serialization.Encoding.PEM)
    )

    print("\nPathSnap certificates generated successfully.")
    print("Private CA key: certificates/ca.key.pem")
    print("Public CA certificate: certificates/ca.crt.pem")
    print("Server key: certificates/server.key.pem")
    print("Server certificate: certificates/server.crt.pem")
    print("\nKeep all private keys confidential.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip",
        help="Optional LAN IPv4 address for future deployment"
    )

    args = parser.parse_args()

    generate_certificates(args.ip)

