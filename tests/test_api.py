import os
import uuid
from datetime import timedelta

import pytest

os.environ["DATABASE_URL"] = "sqlite:///./test_kubaho.db"

from fastapi.testclient import TestClient  # noqa: E402

from api import services  # noqa: E402
from api.db import SessionLocal, engine, utcnow  # noqa: E402
from api.main import app  # noqa: E402
from api.models import Alert, SmsLog  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        assert c.post("/demo/seed", params={"reset": True}).json()["status"] == "seeded"
        yield c
    engine.dispose()
    if os.path.exists("test_kubaho.db"):
        os.remove("test_kubaho.db")


def fall(client, node="node-hh002-a", **extra):
    body = {"event_id": str(uuid.uuid4()), "node_id": node, "type": "fall", "confidence": 0.9, **extra}
    return client.post("/events", json=body)


def test_seed_creates_households_and_restlessness_alert(client):
    hh = {h["id"]: h for h in client.get("/households").json()}
    assert set(hh) == {"HH-001", "HH-002", "HH-003"}
    alerts = client.get("/households/HH-002/alerts").json()
    assert any(a["type"] == "restlessness" and a["severity"] == "non_urgent" for a in alerts)
    nights = client.get("/households/HH-002/restlessness").json()
    assert [n["flagged"] for n in nights[-3:]] == [True, True, True]


def test_fall_event_raises_one_alert_and_sms(client):
    r = fall(client)
    assert r.status_code == 200 and r.json()["alert_id"]
    alert_id = r.json()["alert_id"]
    # A second fall within the de-duplication window does not raise a second alert.
    assert fall(client).json()["alert_id"] is None
    sms = client.get("/households/HH-002/sms").json()
    assert any(s["alert_id"] == alert_id and s["recipient_role"] == "caregiver" for s in sms)


def test_duplicate_event_id_is_ignored(client):
    body = {"event_id": "dup-1", "node_id": "node-hh003-a", "type": "heartbeat"}
    assert client.post("/events", json=body).json()["duplicate"] is False
    assert client.post("/events", json=body).json()["duplicate"] is True


def test_unknown_node_rejected(client):
    assert fall(client, node="nope").status_code == 422


def test_escalation_after_five_minutes_then_sms_ack(client):
    with SessionLocal() as db:
        alert = Alert(household_id="HH-003", type="fall", severity="urgent", created_at=utcnow() - timedelta(minutes=6))
        db.add(alert)
        db.commit()
        escalated = services.escalate_overdue(db)
        assert alert.id in [a.id for a in escalated]
        chw_sms = db.query(SmsLog).filter_by(alert_id=alert.id, recipient_role="chw").count()
        assert chw_sms == 1
        alert_id = alert.id
    r = client.post("/sms/incoming", data={"from": "+250788000190", "text": "2"})
    assert r.json()["matched_alert"] == alert_id
    with SessionLocal() as db:
        a = db.get(Alert, alert_id)
        assert a.status == "acknowledged" and a.ack_outcome == "false_alarm"


def test_bathroom_dwell_alert(client):
    body = {"event_id": str(uuid.uuid4()), "node_id": "node-hh001-b", "type": "bathroom_enter",
            "ts": (utcnow() - timedelta(minutes=35)).isoformat()}
    client.post("/events", json=body)
    with SessionLocal() as db:
        alerts = services.check_bathroom_dwell(db)
    assert [a.type for a in alerts] == ["bathroom_dwell"]
    summary = client.get("/households/HH-001/bathroom").json()
    assert summary["visits"][0]["in_progress"] and summary["visits"][0]["over_threshold"]


def test_dashboard_ack(client):
    alert_id = fall(client, node="node-hh003-a").json()["alert_id"]
    r = client.post(f"/alerts/{alert_id}/ack", json={"outcome": "confirmed", "by": "test"})
    assert r.json()["status"] == "acknowledged"


def test_mqtt_worker_handles_events_and_last_will(client):
    import json
    from types import SimpleNamespace

    from api.worker import on_message

    payload = {"event_id": str(uuid.uuid4()), "type": "fall", "confidence": 0.88}
    on_message(None, None, SimpleNamespace(topic="kubaho/node-hh002-a/events", payload=json.dumps(payload).encode()))
    on_message(None, None, SimpleNamespace(topic="kubaho/node-hh001-b/status", payload=b"offline"))
    nodes = {n["id"]: n["status"] for n in client.get("/households/HH-001").json()["nodes"]}
    assert nodes["node-hh001-b"] == "offline"
    with SessionLocal() as db:
        assert db.query(Alert).filter_by(household_id="HH-001", type="node_offline").count() >= 1
    # A malformed message is logged and dropped, not raised.
    on_message(None, None, SimpleNamespace(topic="kubaho/node-hh002-a/events", payload=b"not json"))


def test_sample_window_shape(client):
    w = client.get("/demo/sample-window", params={"activity": "kneel_pray", "seed": 1}).json()
    assert len(w["csi_amplitude"]) == 200 and len(w["csi_amplitude"][0]) == 52
