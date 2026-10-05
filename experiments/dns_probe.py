import socket, time, statistics
def t(host, n=10):
    xs=[]
    for _ in range(n):
        t0=time.perf_counter()
        socket.getaddrinfo(host, 5173, proto=socket.IPPROTO_TCP)
        xs.append((time.perf_counter()-t0)*1000)
    return statistics.mean(xs), min(xs), max(xs)
for h in ["localhost","127.0.0.1","::1"]:
    try:
        m,a,b = t(h)
        print(f"{h:12} getaddrinfo mean={m:8.1f}ms min={a:7.1f} max={b:7.1f}")
    except Exception as e:
        print(f"{h:12} ERROR {e}")
print()
print("resolved localhost ->", socket.getaddrinfo("localhost", 5173, proto=socket.IPPROTO_TCP)[0][4])
