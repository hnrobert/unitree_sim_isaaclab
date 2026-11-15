# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Four separate camera windows (outside the main viewport).

Subscribes to the running simulator's teleimager ZMQ JPEG streams
(the container is on the host network, ports are local):

  55555  head stereo, side-by-side 1280x480 -> split into L / R
  55556  left wrist camera
  55557  right wrist camera

Each SUB socket uses CONFLATE so only the newest frame is kept -
the windows follow the live view with bounded latency. Run it via
`./run.sh cameras` (docker exec, so the X11 windows land on the
host desktop next to the Isaac viewport).
"""

import argparse
import time

import cv2
import numpy as np
import zmq

WINDOWS = (
    # title, port, half (None = full frame, "L"/"R" = split stereo)
    ("head left", 55555, "L"),
    ("head right", 55555, "R"),
    ("wrist left", 55556, None),
    ("wrist right", 55557, None),
)


def _subscribe(ctx, port):
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.CONFLATE, 1)  # keep only the newest frame
    s.setsockopt(zmq.SUBSCRIBE, b"")  # raw JPEG, no topic prefix
    s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://127.0.0.1:{port}")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640, help="per-view width")
    ap.add_argument("--scale", type=float, default=1.0, help="extra window scale")
    ap.add_argument("--grid", action="store_true",
                    help="single 2x2 grid window instead of four windows")
    args = ap.parse_args()

    ctx = zmq.Context()
    subs = {}
    frames = {t: None for t, _, _ in WINDOWS}
    for title, port, _half in WINDOWS:
        if port not in subs:
            subs[port] = _subscribe(ctx, port)

    w = int(args.width * args.scale)
    h = int(w * 480 / 640)
    if args.grid:
        cv2.namedWindow("cameras 2x2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cameras 2x2", 2 * w, 2 * h)
        cv2.moveWindow("cameras 2x2", 60, 60)
    else:
        for i, (title, _p, _half) in enumerate(WINDOWS):
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, w, h)
            cv2.moveWindow(title, 40 + (i % 2) * (w + 40), 60 + (i // 2) * (h + 80))

    last_print = time.time()
    shown = 0
    while True:
        for port, sock in subs.items():
            try:
                data = sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            half = next(h for t, p, h in WINDOWS if p == port)
            if half == "L":
                frames["head left"] = img[:, : img.shape[1] // 2]
                frames["head right"] = img[:, img.shape[1] // 2:]
            else:
                for t, p, hf in WINDOWS:
                    if p == port and hf is None:
                        frames[t] = img
        shown = 0
        if args.grid:
            tiles = []
            for row in (("head left", "head right"), ("wrist left", "wrist right")):
                cells = []
                for title in row:
                    f = frames[title]
                    if f is None:
                        f = np.zeros((h, w, 3), dtype=np.uint8)
                    else:
                        f = cv2.resize(f, (w, h))
                    cv2.putText(f, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cells.append(f)
                    if frames[title] is not None:
                        shown += 1
                tiles.append(np.hstack(cells))
            cv2.imshow("cameras 2x2", np.vstack(tiles))
        else:
            for title, _p, _half in WINDOWS:
                f = frames[title]
                if f is not None:
                    cv2.imshow(title, f)
                    shown += 1
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        now = time.time()
        if now - last_print > 10.0:
            print(f"[camera_windows] {shown}/4 views live", flush=True)
            last_print = now
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
