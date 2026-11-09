"""Demo data: three fictional households in Bumbogo Sector with two weeks of history.

All names and phone numbers are made up. Run with `python -m api.seed` or POST /demo/seed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import delete, insert
from sqlalchemy.orm import Session

from . import services
from .config import settings
from .db import Base, SessionLocal, init_db, utcnow
from .models import Alert, BathroomVisit, Household, MotionMinute, Node, Person, SmsLog
from .sms import render

HOUSEHOLDS = [
    {
        "id": "HH-001", "name": "Mukamana household", "cell": "Ngara", "village": "Ngara",
        "people": [("elder", "Speciose Mukamana", None, "rw"),
                   ("caregiver", "Jean Bosco Niyonzima (son)", "+250788000101", "rw"),
                   ("caregiver", "Aline Uwase (granddaughter)", "+250788000102", "en"),
                   ("chw", "Odette Mukeshimana (CHW)", "+250788000190", "rw")],
        "nodes": [("node-hh001-a", "living_room", "online"), ("node-hh001-b", "bathroom", "online")],
    },
    {
        "id": "HH-002", "name": "Habimana household", "cell": "Musave", "village": "Musave",
        "people": [("elder", "Evariste Habimana", None, "rw"),
                   ("caregiver", "Claudine Ingabire (daughter)", "+250788000201", "rw"),
                   ("chw", "Emmanuel Nsengimana (CHW)", "+250788000290", "rw")],
        "nodes": [("node-hh002-a", "living_room", "online")],
    },
    {
        "id": "HH-003", "name": "Nyirabahire household", "cell": "Kinyaga", "village": "Kinyaga",
        "people": [("elder", "Agnes Nyirabahire", None, "rw"),
                   ("caregiver", "Patrick Mugisha (nephew)", "+250788000301", "en"),
                   ("chw", "Odette Mukeshimana (CHW)", "+250788000190", "rw")],
        "nodes": [("node-hh003-a", "living_room", "offline")],
    },
]

# Restless minutes per night: (typical mean, extra minutes on the last three nights)
RESTLESS_PROFILE = {"HH-001": (38, 0), "HH-002": (45, 75), "HH-003": (30, 0)}


def _night_start_utc(night_end: datetime) -> datetime:
    """22:00 local the evening before `night_end` (a local date at 00:00), in UTC."""
    return night_end - timedelta(hours=2) - timedelta(hours=settings.tz_offset_hours)


def seed(db: Session, reset: bool = False) -> dict:
    if reset:
        for table in reversed(Base.metadata.sorted_tables):
            db.execute(delete(table))
        db.commit()
    elif db.get(Household, "HH-001"):
        return {"status": "exists", "hint": "pass reset=true to recreate the demo data"}

    rng = np.random.default_rng(2026)
    now = utcnow()
    today_local = (now + timedelta(hours=settings.tz_offset_hours)).replace(hour=0, minute=0, second=0, microsecond=0)
    last_seen_offline = now - timedelta(hours=3)

    for spec in HOUSEHOLDS:
        hh = Household(id=spec["id"], name=spec["name"], cell=spec["cell"], village=spec["village"])
        hh.people = [Person(role=r, name=n, phone=p, language=lang) for r, n, p, lang in spec["people"]]
        hh.nodes = [Node(id=i, zone=z, status=s, last_seen=now if s == "online" else last_seen_offline)
                    for i, z, s in spec["nodes"]]
        db.add(hh)
    db.commit()

    # 14 nights of per-minute motion energy (22:00-06:00).
    rows = []
    for hh_id, (typical, extra) in RESTLESS_PROFILE.items():
        for k in range(13, -1, -1):
            start = _night_start_utc(today_local - timedelta(days=k))
            target = max(0, int(rng.normal(typical, 6)) + (extra + int(rng.normal(0, 8)) if k < 3 else 0))
            restless = set(rng.choice(480, size=min(target, 480), replace=False).tolist())
            for m in range(480):
                energy = rng.uniform(0.32, 0.9) if m in restless else rng.uniform(0.0, 0.25)
                rows.append({"household_id": hh_id, "ts": start + timedelta(minutes=m), "energy": round(float(energy), 3)})
    db.execute(insert(MotionMinute), rows)

    # A week of bathroom visits, one of them too long (HH-003, four days ago).
    for hh_id in RESTLESS_PROFILE:
        for d in range(7, 0, -1):
            day = today_local - timedelta(days=d) - timedelta(hours=settings.tz_offset_hours)
            hours = sorted(rng.choice([1, 3, 4, 7, 9, 12, 14, 17, 19, 21, 23], size=rng.integers(4, 8), replace=False))
            for h in hours:
                enter = day + timedelta(hours=int(h), minutes=int(rng.integers(0, 59)))
                minutes = 41.0 if (hh_id == "HH-003" and d == 4 and h == hours[-1]) else float(rng.uniform(2, 9))
                db.add(BathroomVisit(household_id=hh_id, enter_ts=enter, exit_ts=enter + timedelta(minutes=minutes),
                                     alerted=minutes > settings.bathroom_dwell_minutes))
    db.commit()

    # Alert history with acknowledgements (field ground truth).
    def past_alert(hh_id, kind, severity, ago, outcome, by, escalated=False, confidence=None):
        created = now - ago
        alert = Alert(household_id=hh_id, type=kind, severity=severity, confidence=confidence, created_at=created,
                      status="acknowledged", ack_outcome=outcome, ack_by=by,
                      ack_at=created + timedelta(minutes=7 if escalated else 2),
                      escalated_at=created + timedelta(minutes=5) if escalated else None, details={"seeded": True})
        db.add(alert)
        db.flush()
        hh = db.get(Household, hh_id)
        values = {"elder": next(p.name for p in hh.people if p.role == "elder"), "village": hh.village,
                  "time": (created + timedelta(hours=settings.tz_offset_hours)).strftime("%H:%M"),
                  "minutes": int(settings.bathroom_dwell_minutes if kind == "bathroom_dwell" else settings.escalation_minutes)}
        for role in ["caregiver"] + (["chw"] if escalated else []):
            template = "escalation" if role == "chw" else kind
            for person in [p for p in hh.people if p.role == role and p.phone]:
                db.add(SmsLog(alert_id=alert.id, household_id=hh_id, phone=person.phone, recipient_role=role,
                              message=render(template, person.language, **values), status="logged",
                              created_at=created + (timedelta(minutes=5) if role == "chw" else timedelta(seconds=4))))
        replier = next((p for p in hh.people if p.phone and by == f"{p.name.split(' (')[0]} (SMS)"), None)
        if replier:  # the acknowledgement arrived as an SMS reply
            db.add(SmsLog(household_id=hh_id, direction="in", phone=replier.phone, recipient_role=replier.role,
                          message="1" if outcome == "confirmed" else "2", status="received",
                          created_at=alert.ack_at))

    past_alert("HH-001", "fall", "urgent", timedelta(days=6, hours=3), "false_alarm", "Jean Bosco Niyonzima (SMS)", confidence=0.71)
    past_alert("HH-001", "fall", "urgent", timedelta(days=2, hours=9), "confirmed", "Aline Uwase (dashboard)", confidence=0.96)
    past_alert("HH-003", "bathroom_dwell", "urgent", timedelta(days=4, hours=2), "confirmed",
               "Odette Mukeshimana (SMS)", escalated=True)
    db.add(Alert(household_id="HH-003", type="node_offline", severity="info", created_at=last_seen_offline,
                 details={"node_id": "node-hh003-a"}))
    db.commit()

    # A live fall alert through the real ingest path (sends SMS in console mode).
    services.ingest_event(db, {"event_id": str(uuid.uuid4()), "node_id": "node-hh001-a", "type": "fall",
                               "ts": now - timedelta(minutes=2), "confidence": 0.93})
    restless = [hh for hh in RESTLESS_PROFILE if services.check_restlessness(db, hh)]
    return {"status": "seeded", "households": [h["id"] for h in HOUSEHOLDS], "restlessness_alerts": restless}


if __name__ == "__main__":
    import sys

    init_db()
    with SessionLocal() as session:
        print(seed(session, reset="--reset" in sys.argv))
