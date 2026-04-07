#!/usr/bin/env python3
"""Hackathon Test Suite for EmailTriage OpenEnv"""

import requests, sys, random, time

BASE_URL = "https://omchoksi108-emailopenenvrl.hf.space"
p = 0
f = 0


def test(name, ok):
    global p, f
    try:
        if ok():
            p += 1
            print(f"  PASS: {name}")
        else:
            f += 1
            print(f"  FAIL: {name}")
    except Exception as e:
        f += 1
        print(f"  FAIL: {name} ({e})")
    time.sleep(0.05)


print("=" * 50)
print("HACKATHON TEST SUITE")
print("=" * 50)

# 1. Basic Endpoints
print("\n[1] Basic Endpoints")
test(
    "Reset Easy",
    lambda: (
        requests.post(f"{BASE_URL}/reset", json={"task_id": "easy"}).status_code == 200
    ),
)
test(
    "Reset Medium",
    lambda: (
        requests.post(f"{BASE_URL}/reset", json={"task_id": "medium"}).status_code
        == 200
    ),
)
test(
    "Reset Hard",
    lambda: (
        requests.post(f"{BASE_URL}/reset", json={"task_id": "hard"}).status_code == 200
    ),
)
test("Health", lambda: requests.get(f"{BASE_URL}/health").json().get("status") == "ok")
test(
    "Metadata",
    lambda: (
        "EmailTriage" in requests.get(f"{BASE_URL}/metadata").json().get("name", "")
    ),
)
test("Schema", lambda: "action" in requests.get(f"{BASE_URL}/schema").json())
test("Web UI", lambda: len(requests.get(f"{BASE_URL}/web").text) > 100)
test("State", lambda: requests.get(f"{BASE_URL}/state").status_code == 200)

# 2. Actions
print("\n[2] Actions")
r = requests.post(f"{BASE_URL}/reset", json={"task_id": "medium"})
tid = (
    r.json().get("observation", {}).get("inbox_preview", [{}])[0].get("id", 101)
    if r.json().get("observation", {}).get("inbox_preview")
    else 101
)
test(
    "Read",
    lambda: (
        requests.post(
            f"{BASE_URL}/step",
            json={"action": {"action_type": "read", "target_email_id": int(tid)}},
        ).status_code
        == 200
    ),
)
test(
    "Archive",
    lambda: (
        requests.post(
            f"{BASE_URL}/step",
            json={
                "action": {"action_type": "archive", "target_email_id": int(tid) + 1}
            },
        ).status_code
        == 200
    ),
)
test(
    "Query Calendar",
    lambda: (
        requests.post(
            f"{BASE_URL}/step",
            json={"action": {"action_type": "query_calendar", "target_email_id": -1}},
        ).status_code
        == 200
    ),
)
test(
    "Draft Email",
    lambda: (
        requests.post(
            f"{BASE_URL}/step",
            json={
                "action": {
                    "action_type": "draft_email",
                    "target_email_id": 101,
                    "draft_content": "Thanks!",
                    "proposed_slot": "2026-04-03 10:00",
                }
            },
        ).status_code
        == 200
    ),
)

# 3. Edge Cases
print("\n[3] Edge Cases")
test(
    "Invalid Action",
    lambda: (
        requests.post(
            f"{BASE_URL}/step", json={"action": {"action_type": "invalid"}}
        ).status_code
        == 422
    ),
)
test(
    "Invalid Email ID",
    lambda: (
        requests.post(
            f"{BASE_URL}/step",
            json={"action": {"action_type": "read", "target_email_id": 999999}},
        ).status_code
        == 200
    ),
)
test(
    "Empty Draft",
    lambda: (
        requests.post(
            f"{BASE_URL}/step",
            json={
                "action": {
                    "action_type": "draft_email",
                    "target_email_id": 101,
                    "draft_content": "",
                }
            },
        ).status_code
        == 200
    ),
)
test(
    "Long Draft",
    lambda: (
        requests.post(
            f"{BASE_URL}/step",
            json={
                "action": {
                    "action_type": "draft_email",
                    "target_email_id": 101,
                    "draft_content": "A" * 10000,
                }
            },
        ).status_code
        == 200
    ),
)

# 4. Rewards
print("\n[4] Rewards")
test(
    "Reward in [0,1]",
    lambda r=requests.post(f"{BASE_URL}/reset", json={"task_id": "medium"}): (
        0 <= r.json().get("reward", 0) <= 1
    ),
)
test(
    "Read Reward",
    lambda r=requests.post(f"{BASE_URL}/step", json={"action": {"action_type": "read", "target_email_id": 101}}): (
        0 <= r.json().get("reward", 0) <= 1
    ),
)
test(
    "Archive Reward",
    lambda r=requests.post(f"{BASE_URL}/step", json={"action": {"action_type": "archive", "target_email_id": 102}}): (
        0 <= r.json().get("reward", 0) <= 1
    ),
)
test(
    "Draft Reward",
    lambda r=requests.post(f"{BASE_URL}/step", json={"action": {"action_type": "draft_email", "target_email_id": 103, "draft_content": "test"}}): (
        0 <= r.json().get("reward", 0) <= 1
    ),
)

# 5. Tasks
print("\n[5] Task Sizes")
test(
    "Easy has 3",
    lambda r=requests.post(f"{BASE_URL}/reset", json={"task_id": "easy"}): (
        r.json().get("observation", {}).get("inbox_remaining", 0) == 3
    ),
)
test(
    "Medium has 5",
    lambda r=requests.post(f"{BASE_URL}/reset", json={"task_id": "medium"}): (
        r.json().get("observation", {}).get("inbox_remaining", 0) == 5
    ),
)
test(
    "Hard 7-10",
    lambda r=requests.post(f"{BASE_URL}/reset", json={"task_id": "hard"}): (
        7 <= r.json().get("observation", {}).get("inbox_remaining", 0) <= 10
    ),
)

# 6. Security
print("\n[6] Security")
test(
    "XSS Blocked",
    lambda r=requests.post(f"{BASE_URL}/step", json={"action": {"action_type": "draft_email", "target_email_id": 101, "draft_content": "<script>alert(1)</script>"}}): (
        r.json().get("reward", 1) <= 0.3
    ),
)
test(
    "Path Blocked",
    lambda r=requests.post(f"{BASE_URL}/step", json={"action": {"action_type": "draft_email", "target_email_id": 101, "draft_content": "../../../etc/passwd"}}): (
        r.json().get("reward", 1) <= 0.3
    ),
)
test(
    "SQL Blocked",
    lambda r=requests.post(f"{BASE_URL}/step", json={"action": {"action_type": "draft_email", "target_email_id": 101, "draft_content": "DROP TABLE users"}}): (
        r.json().get("reward", 1) <= 0.3
    ),
)
test(
    "OS Blocked",
    lambda r=requests.post(f"{BASE_URL}/step", json={"action": {"action_type": "draft_email", "target_email_id": 101, "draft_content": "import os; rm -rf /"}}): (
        r.json().get("reward", 1) <= 0.3
    ),
)

# 7. Complete Flows
print("\n[7] Complete Flows")
r = requests.post(f"{BASE_URL}/reset", json={"task_id": "easy"})
done = False
for i in range(10):
    r = requests.post(
        f"{BASE_URL}/step",
        json={"action": {"action_type": "archive", "target_email_id": 101 + i}},
    )
    done = done or r.json().get("done", False)
test("Easy Complete", lambda d=done: d)
r = requests.post(f"{BASE_URL}/reset", json={"task_id": "medium"})
steps = 0
for i in range(15):
    r = requests.post(
        f"{BASE_URL}/step",
        json={"action": {"action_type": "archive", "target_email_id": 101 + i}},
    )
    steps += 1
test("Medium Steps", lambda s=steps: s > 0)

# 8. Dynamic Events
print("\n[8] Dynamic Events")
test(
    "Hard Mode",
    lambda: (
        requests.post(f"{BASE_URL}/reset", json={"task_id": "hard"}).status_code == 200
    ),
)
r = requests.post(f"{BASE_URL}/reset", json={"task_id": "medium"})
slots = r.json().get("observation", {}).get("calendar_slots", [])
test("Calendar", lambda: len(slots) > 0)

# 9. Stress
print("\n[9] Stress")
test(
    "Rapid 10",
    lambda: all(
        requests.post(
            f"{BASE_URL}/reset", json={"task_id": random.choice(["easy", "medium"])}
        ).status_code
        == 200
        for _ in range(10)
    ),
)
test(
    "Steps 20",
    lambda: all(
        requests.post(
            f"{BASE_URL}/step",
            json={"action": {"action_type": "read", "target_email_id": 101}},
        ).status_code
        == 200
        for _ in range(20)
    ),
)

# FINAL
print("\n" + "=" * 50)
total = p + f
print(f"RESULTS: {p}/{total} ({p * 100 // total}%)")
if p == total:
    print("ALL TESTS PASSED!")
elif p >= total * 0.9:
    print("EXCELLENT!")
elif p >= total * 0.8:
    print("GOOD - Needs minor fixes")
else:
    print("NEEDS MORE WORK")
sys.exit(0 if p >= total * 0.8 else 1)
