
from networking.discovery_security import (
    create_announcement,
    verify_announcement
)


print("\nPATHSNAP DISCOVERY SECURITY TEST\n")

# Test 1: Generate announcement
announcement = create_announcement(
    "127.0.0.1",
    8000
)

print("1. Announcement generated")

# Test 2: Verify authentic message
result = verify_announcement(announcement)

print("2. Valid announcement:", result)

assert result is True

# Test 3: Reject replayed announcement
replay_result = verify_announcement(announcement)

print("3. Replayed announcement:", replay_result)

assert replay_result is False

# Test 4: Reject modified message
tampered = create_announcement(
    "127.0.0.1",
    8000
)

tampered["host"] = "192.168.1.100"

tampered_result = verify_announcement(tampered)

print("4. Tampered announcement:", tampered_result)

assert tampered_result is False

print("\nALL DISCOVERY SECURITY TESTS PASSED")
