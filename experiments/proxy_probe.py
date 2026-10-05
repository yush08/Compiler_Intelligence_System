import urllib.request, os, time, statistics, json, socket
print("getproxies():", urllib.request.getproxies())
for h in ["localhost","127.0.0.1","::1"]:
    try:
        print(f"  proxy_bypass({h}) =", urllib.request.proxy_bypass(h))
    except Exception as e:
        print(f"  proxy_bypass({h}) ERROR {e}")
print("env:", {k:v for k,v in os.environ.items() if "proxy" in k.lower()})

# same test, proxies explicitly disabled
tac = ("k1 = 42\nk2 = k1 + 8\nk3 = input_a + input_b\nk4 = input_a + input_b\n"
       "k5 = k3 * 1\nunused_k = 77\nk6 = k4 + k1")
payload = json.dumps({"tac": tac}).encode()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for base, label in [("http://localhost:5173","localhost, no-proxy"),
                    ("http://127.0.0.1:5173","127.0.0.1, no-proxy"),
                    ("http://[::1]:5173","::1, no-proxy")]:
    xs=[]
    for _ in range(5):
        req = urllib.request.Request(base+"/api/analyze", data=payload,
              headers={"Content-Type":"application/json"}, method="POST")
        t0=time.perf_counter()
        try:
            with opener.open(req, timeout=20) as r: r.read()
        except Exception as e:
            print("   err", type(e).__name__, e); continue
        xs.append((time.perf_counter()-t0)*1000)
    if xs: print(f"{label:24} mean={statistics.mean(xs):8.1f}ms  min={min(xs):7.1f} max={max(xs):7.1f}")
