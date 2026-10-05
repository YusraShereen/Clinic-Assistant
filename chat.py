import json, sys, urllib.request
text = " ".join(sys.argv[1:])
req = urllib.request.Request("http://localhost:8000/chat",
    data=json.dumps({"text": text, "session_id": "demo"}).encode("utf-8"),
    headers={"Content-Type": "application/json"})
out = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
print(json.dumps(out, ensure_ascii=False, indent=2))