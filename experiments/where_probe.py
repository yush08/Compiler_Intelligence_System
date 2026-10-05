import socket, time, statistics, json, urllib.request

def connect_ms(host, port, n=5):
    xs=[]
    for _ in range(n):
        t0=time.perf_counter()
        try:
            s=socket.create_connection((host,port), timeout=10); s.close()
        except Exception as e:
            return None, type(e).__name__
        xs.append((time.perf_counter()-t0)*1000)
    return statistics.mean(xs), (min(xs), max(xs))

for h in ["localhost","127.0.0.1"]:
    print(f"raw TCP connect {h:12}: {connect_ms(h,5173)}")

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
tac = "k1 = 42\nk2 = k1 + 8\nunused_k = 77"
payload = json.dumps({"tac": tac}).encode()

def timed(method, path, data=None):
    xs=[]
    for _ in range(5):
        req = urllib.request.Request("http://localhost:5173"+path, data=data,
              headers={"Content-Type":"application/json"} if data else {}, method=method)
        t0=time.perf_counter()
        with opener.open(req, timeout=20) as r: r.read()
        xs.append((time.perf_counter()-t0)*1000)
    return round(statistics.mean(xs),1)

print()
print("localhost  GET  /            (static) :", timed("GET","/"), "ms")
print("localhost  GET  /api/health   (proxied):", timed("GET","/api/health"), "ms")
print("localhost  POST /api/analyze  (proxied):", timed("POST","/api/analyze",payload), "ms")
