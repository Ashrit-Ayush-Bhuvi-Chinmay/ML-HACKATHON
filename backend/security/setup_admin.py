
import argparse

from security.device_store import (
    initialize_database,
    promote_to_admin
)


def main():
    parser = argparse.ArgumentParser(
        description="Promote a registered PathSnap device to admin"
    )

    parser.add_argument(
        "--device-id",
        required=True,
        help="UUID of your registered device"
    )

    args = parser.parse_args()

    initialize_database()

    success = promote_to_admin(args.device_id)

    if success:
        print("Administrator role assigned successfully.")
    else:
        print("Device not found or already revoked.")


if __name__ == "__main__":
    main()
