import json, time, statistics, urllib.request, urllib.error

tac = ("k1 = 42\nk2 = k1 + 8\nk3 = input_a + input_b\nk4 = input_a + input_b\n"
       "k5 = k3 * 1\nunused_k = 77\nk6 = k4 + k1")
payload = json.dumps({"tac": tac}).encode()

targets = [
    ("backend-direct 127.0.0.1:8000", "http://127.0.0.1:8000", 5),
    ("proxy localhost:5173",           "http://localhost:5173",  5),
    ("proxy 127.0.0.1:5173",           "http://127.0.0.1:5173",  5),
]
for label, base, n in targets:
    times = []
    codes = []
    for i in range(n):
        req = urllib.request.Request(base + "/api/analyze", data=payload,
              headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                r.read(); codes.append(r.status)
        except urllib.error.HTTPError as e:
            codes.append(e.code)
        except Exception as e:
            codes.append("ERR:" + type(e).__name__)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"{label:32} codes={codes} mean={statistics.mean(times):8.1f}ms min={min(times):8.1f} max={max(times):8.1f}")
