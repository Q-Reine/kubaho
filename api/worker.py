"""Alert worker: consumes node events from MQTT, escalates unacknowledged alerts to the CHW,
watches bathroom dwell time and checks night-time restlessness.

Run with `python -m api.worker`.
"""

from __future__ import annotations

import json
import logging
import signal
import ssl
import threading
import time

from sqlalchemy import select

from . import services
from .config import settings
from .db import SessionLocal, init_db
from .models import Household

log = logging.getLogger("kubaho.worker")
TICK_SECONDS = 15
RESTLESS_EVERY_SECONDS = 600


def on_message(client, userdata, msg) -> None:
    # Topics: kubaho/<node_id>/events (JSON) and kubaho/<node_id>/status ("online" / "offline", RuView last-will)
    parts = msg.topic.split("/")
    if len(parts) != 3:
        return
    _, node_id, kind = parts
    with SessionLocal() as db:
        try:
            if kind == "events":
                payload = json.loads(msg.payload)
                payload.setdefault("node_id", node_id)
                event, alert = services.ingest_event(db, payload)
                if alert:
                    log.warning("alert %s raised from MQTT event %s", alert.id, payload.get("event_id"))
            elif kind == "status":
                services.set_node_status(db, node_id, msg.payload.decode().strip().lower())
        except (services.IngestError, json.JSONDecodeError, KeyError) as exc:
            log.error("rejected message on %s: %s", msg.topic, exc)


def start_mqtt():
    import paho.mqtt.client as mqtt

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="kubaho-worker")
    if settings.mqtt_username:
        client.username_pw_set(settings.mqtt_username, settings.mqtt_password)
    if settings.mqtt_tls:
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)

    def on_connect(c, userdata, flags, reason_code, properties):
        log.info("MQTT connected (%s); subscribing", reason_code)
        c.subscribe([("kubaho/+/events", 1), ("kubaho/+/status", 1)])

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(settings.mqtt_host, settings.mqtt_port, keepalive=30)
    client.loop_start()
    return client


def tick(check_restlessness: bool) -> None:
    with SessionLocal() as db:
        services.escalate_overdue(db)
        services.check_bathroom_dwell(db)
        if check_restlessness:
            for hh_id in db.scalars(select(Household.id)):
                services.check_restlessness(db, hh_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()
    client = start_mqtt() if settings.mqtt_host else None
    if client is None:
        log.info("MQTT_HOST not set: only escalation / dwell / restlessness checks will run")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    last_restless = 0.0
    while not stop.is_set():
        due = time.time() - last_restless > RESTLESS_EVERY_SECONDS
        try:
            tick(check_restlessness=due)
            if due:
                last_restless = time.time()
        except Exception:  # keep the worker alive; the next tick retries
            log.exception("worker tick failed")
        stop.wait(TICK_SECONDS)
    if client:
        client.loop_stop()


if __name__ == "__main__":
    main()
