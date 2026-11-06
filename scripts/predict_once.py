"""Send a single prediction request to the running serve endpoint."""

from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--features", default="0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")
    args = parser.parse_args()

    features = [float(x) for x in args.features.split(",")]
    body = json.dumps({"features": features}).encode("utf-8")
    req = urllib.request.Request(
        f"{args.url}/predict", data=body, headers={"Content-Type": "application/json"}
    )
    # Bypass any system proxy so localhost requests are not intercepted.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=10) as resp:
        print(resp.read().decode("utf-8"))


if __name__ == "__main__":
    main()
