"""Simulated Kubaho sensing node.

Generates synthetic CSI windows for a sequence of activities, classifies each one with the
deployed model (POST /predict - standing in for on-device inference on the ESP32-S3) and
publishes the resulting events over MQTT, or over HTTP when no broker is given.

Examples:
    python simulator/node_sim.py --scenario demo                      # HTTP to localhost:8000
    python simulator/node_sim.py --scenario demo --mqtt-host localhost
    python simulator/node_sim.py --scenario bathroom --node node-hh001-b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kubaho.synthetic import KUBAHO_ACTIVITIES, make_layout, make_subject, simulate_clip  # noqa: E402

DEMO_SEQUENCE = ["walk", "sit_on_mat", "kneel_pray", "squat_stool", "bend_sweep", "fall_forward", "lie_on_mat", "empty"]


class Publisher:
    def __init__(self, node_id: str, api: str, mqtt_host: str | None, mqtt_port: int):
        self.node_id, self.api = node_id, api.rstrip("/")
        self.mqtt = None
        if mqtt_host:
            import paho.mqtt.client as mqtt

            self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim-{node_id}")
            # RuView-style availability: the broker publishes "offline" if the node disappears.
            self.mqtt.will_set(f"kubaho/{node_id}/status", "offline", qos=1, retain=True)
            self.mqtt.connect(mqtt_host, mqtt_port, keepalive=30)
            self.mqtt.loop_start()
            self.mqtt.publish(f"kubaho/{node_id}/status", "online", qos=1, retain=True)

    def event(self, etype: str, ts: datetime | None = None, **fields) -> None:
        payload = {"event_id": str(uuid.uuid4()), "node_id": self.node_id, "type": etype,
                   "ts": (ts or datetime.now(timezone.utc)).isoformat(), **fields}
        if self.mqtt:
            self.mqtt.publish(f"kubaho/{self.node_id}/events", json.dumps(payload), qos=1).wait_for_publish()
            print(f"  -> MQTT {etype} {fields}")
        else:
            r = httpx.post(f"{self.api}/events", json=payload, timeout=30)
            r.raise_for_status()
            print(f"  -> HTTP {etype} {fields} {r.json()}")

    def close(self) -> None:
        if self.mqtt:
            self.mqtt.loop_stop()
            self.mqtt.disconnect()


def classify(api: str, window: np.ndarray) -> dict:
    r = httpx.post(f"{api}/predict", json={"csi_amplitude": window.round(4).tolist(), "fs": 100}, timeout=30)
    r.raise_for_status()
    return r.json()


def run_demo(pub: Publisher, api: str, interval: float, loops: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    layout = make_layout("pilot-home", 999, rural=True)
    subject = make_subject(rng)
    for i in range(loops):
        for activity in DEMO_SEQUENCE:
            csi = simulate_clip(activity, layout, subject, rng)
            window = np.nan_to_num(np.abs(csi[50:250]))
            result = classify(api, window)
            mark = "OK " if result["predicted_class"] == KUBAHO_ACTIVITIES[activity] else "ERR"
            print(f"[{mark}] actual={activity:<14} predicted={result['predicted_class']:<13} "
                  f"P(fall)={result['fall_probability']:.2f} alert={result['fall_alert']}")
            energy = float(np.abs(np.diff(window, axis=0)).mean())
            pub.event("motion_summary", energy=round(min(energy, 1.0), 3))
            if result["fall_alert"]:
                pub.event("fall", confidence=result["fall_probability"])
            time.sleep(interval)


def run_bathroom(pub: Publisher, minutes_ago: float) -> None:
    """An elder entered the bathroom `minutes_ago` and has not come out: the worker should raise an alert."""
    pub.event("bathroom_enter", ts=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago))
    print("bathroom_enter sent; the worker raises a dwell alert within ~15 s if the visit exceeds 30 min")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--mqtt-host")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--node", default="node-hh001-a")
    ap.add_argument("--scenario", choices=["demo", "bathroom"], default="demo")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--minutes-ago", type=float, default=35)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    pub = Publisher(args.node, args.api, args.mqtt_host, args.mqtt_port)
    try:
        if args.scenario == "demo":
            run_demo(pub, args.api, args.interval, args.loops, args.seed)
        else:
            run_bathroom(pub, args.minutes_ago)
    finally:
        pub.close()


if __name__ == "__main__":
    main()
