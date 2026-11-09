"""Event ingestion, alerting, escalation and wellness analytics.

Shared by the HTTP API (POST /events) and the MQTT worker so both paths behave the same.
"""

from __future__ import annotations

import logging
import statistics
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import settings
from .db import utcnow
from .models import Alert, BathroomVisit, Event, Household, MotionMinute, Node, Person, SmsLog
from .sms import render, send_sms

log = logging.getLogger("kubaho.services")

EVENT_TYPES = {"fall", "motion_summary", "bathroom_enter", "bathroom_exit", "heartbeat"}


class IngestError(ValueError):
    pass


def local(ts: datetime) -> datetime:
    return ts + timedelta(hours=settings.tz_offset_hours)


def _people(db: Session, household_id: str, role: str) -> list[Person]:
    return list(db.scalars(select(Person).where(Person.household_id == household_id, Person.role == role)))


def _elder_name(db: Session, household_id: str) -> str:
    elders = _people(db, household_id, "elder")
    return elders[0].name if elders else household_id


def notify(db: Session, alert: Alert, role: str, kind: str, **values) -> int:
    """Send a templated SMS to every person with `role` in the alert's household."""
    hh = db.get(Household, alert.household_id)
    values = {"elder": _elder_name(db, alert.household_id), "village": hh.village if hh else "",
              "time": local(alert.created_at).strftime("%H:%M"), **values}
    sent = 0
    for person in _people(db, alert.household_id, role):
        if not person.phone:
            continue
        message = render(kind, person.language, **values)
        status = send_sms(person.phone, message)
        db.add(SmsLog(alert_id=alert.id, household_id=alert.household_id, phone=person.phone,
                      recipient_role=role, message=message[:480], status=status))
        sent += 1
    return sent


def ingest_event(db: Session, payload: dict) -> tuple[Event | None, Alert | None]:
    """Validate, de-duplicate and store one node event; raise alerts where needed.

    Returns (event, alert); event is None for a duplicate delivery.
    """
    etype = payload.get("type")
    if etype not in EVENT_TYPES:
        raise IngestError(f"unknown event type {etype!r}")
    node = db.get(Node, payload.get("node_id"))
    if node is None:
        raise IngestError(f"unknown node {payload.get('node_id')!r}")
    confidence = payload.get("confidence")
    if confidence is not None and not 0 <= confidence <= 1:
        raise IngestError("confidence must be between 0 and 1")
    ts = payload.get("ts") or utcnow()
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

    event = Event(event_uid=str(payload["event_id"]), node_id=node.id, household_id=node.household_id,
                  type=etype, confidence=confidence, ts=ts,
                  payload={k: v for k, v in payload.items() if k not in {"event_id", "node_id", "type", "ts"}})
    db.add(event)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        return None, None

    node.status, node.last_seen = "online", utcnow()
    alert = None
    if etype == "fall":
        alert = _raise_fall_alert(db, node.household_id, ts, confidence, event.id)
    elif etype == "motion_summary":
        db.merge(MotionMinute(household_id=node.household_id, ts=ts.replace(second=0, microsecond=0),
                              energy=float(payload["energy"])))
    elif etype == "bathroom_enter":
        db.add(BathroomVisit(household_id=node.household_id, enter_ts=ts))
    elif etype == "bathroom_exit":
        visit = db.scalars(select(BathroomVisit).where(BathroomVisit.household_id == node.household_id,
                                                       BathroomVisit.exit_ts.is_(None))
                           .order_by(BathroomVisit.enter_ts.desc())).first()
        if visit:
            visit.exit_ts = ts
    db.commit()
    return event, alert


def _raise_fall_alert(db: Session, household_id: str, ts: datetime, confidence: float | None, event_id: int) -> Alert | None:
    recent = db.scalars(select(Alert).where(
        Alert.household_id == household_id, Alert.type == "fall",
        Alert.created_at >= utcnow() - timedelta(seconds=settings.fall_dedupe_seconds))).first()
    if recent:
        return None
    alert = Alert(household_id=household_id, type="fall", severity="urgent", confidence=confidence,
                  details={"event_id": event_id, "event_ts": ts.isoformat()})
    db.add(alert)
    db.flush()
    notify(db, alert, "caregiver", "fall")
    log.warning("fall alert %s for %s (confidence %s)", alert.id, household_id, confidence)
    return alert


def set_node_status(db: Session, node_id: str, status: str) -> None:
    node = db.get(Node, node_id)
    if node is None:
        return
    went_offline = status == "offline" and node.status != "offline"
    node.status = status
    if status == "online":
        node.last_seen = utcnow()
    if went_offline:
        db.add(Alert(household_id=node.household_id, type="node_offline", severity="info",
                     details={"node_id": node_id}))
    db.commit()


def acknowledge(db: Session, alert: Alert, outcome: str, by: str) -> Alert:
    alert.status, alert.ack_outcome, alert.ack_by, alert.ack_at = "acknowledged", outcome, by, utcnow()
    db.commit()
    return alert


def handle_incoming_sms(db: Session, phone: str, text: str) -> Alert | None:
    """Caregiver/CHW replies '1' (confirmed) or '2' (false alarm).

    The reply is matched to the most recent still-open alert this phone number was texted about, so a
    CHW who covers several households acknowledges the right one.
    """
    choice = text.strip()[:1]
    alert = db.scalars(select(Alert).join(SmsLog, SmsLog.alert_id == Alert.id)
                       .where(SmsLog.phone == phone, SmsLog.direction == "out",
                              Alert.status.in_(["open", "escalated"]))
                       .order_by(SmsLog.created_at.desc())).first()
    person = None
    if alert is not None:
        person = db.scalars(select(Person).where(Person.phone == phone,
                                                 Person.household_id == alert.household_id)).first()
    db.add(SmsLog(direction="in", phone=phone, message=text[:480], status="received",
                  household_id=alert.household_id if alert else None,
                  recipient_role=person.role if person else None))
    if alert is None or choice not in {"1", "2"}:
        db.commit()
        return None
    by = f"{person.name.split(' (')[0]} (SMS)" if person else f"{phone} (SMS)"
    return acknowledge(db, alert, "confirmed" if choice == "1" else "false_alarm", by)


def escalate_overdue(db: Session, now: datetime | None = None) -> list[Alert]:
    """Escalate urgent alerts not acknowledged within ESCALATION_MINUTES to the CHW."""
    now = now or utcnow()
    cutoff = now - timedelta(minutes=settings.escalation_minutes)
    overdue = list(db.scalars(select(Alert).where(Alert.status == "open", Alert.severity == "urgent",
                                                  Alert.created_at <= cutoff)))
    for alert in overdue:
        alert.status, alert.escalated_at = "escalated", now
        notify(db, alert, "chw", "escalation", minutes=int(settings.escalation_minutes))
        log.warning("alert %s escalated to CHW", alert.id)
    db.commit()
    return overdue


def check_bathroom_dwell(db: Session, now: datetime | None = None) -> list[Alert]:
    now = now or utcnow()
    cutoff = now - timedelta(minutes=settings.bathroom_dwell_minutes)
    visits = list(db.scalars(select(BathroomVisit).where(BathroomVisit.exit_ts.is_(None),
                                                          BathroomVisit.alerted.is_(False),
                                                          BathroomVisit.enter_ts <= cutoff)))
    alerts = []
    for visit in visits:
        visit.alerted = True
        alert = Alert(household_id=visit.household_id, type="bathroom_dwell", severity="urgent",
                      details={"visit_id": visit.id, "entered": visit.enter_ts.isoformat()})
        db.add(alert)
        db.flush()
        notify(db, alert, "caregiver", "bathroom_dwell", minutes=int(settings.bathroom_dwell_minutes))
        alerts.append(alert)
    db.commit()
    return alerts


def _night_of(ts: datetime) -> date | None:
    """Night 22:00-06:00 local time, labelled by the date it ends."""
    t = local(ts)
    if t.hour >= 22:
        return (t + timedelta(days=1)).date()
    if t.hour < 6:
        return t.date()
    return None


def restlessness(db: Session, household_id: str, nights: int = 14) -> list[dict]:
    """Nightly restlessness index with a z-score against the elder's own 7-night baseline."""
    since = utcnow() - timedelta(days=nights + settings.baseline_nights + 1)
    rows = db.execute(select(MotionMinute.ts, MotionMinute.energy).where(
        MotionMinute.household_id == household_id, MotionMinute.ts >= since)).all()
    counts: dict[date, int] = {}
    for ts, energy in rows:
        night = _night_of(ts)
        if night is not None:
            counts[night] = counts.get(night, 0) + (energy > settings.restless_energy_threshold)
    out = []
    normal: list[int] = []  # the baseline is built from un-flagged nights only, so a streak can't absorb itself
    for night in sorted(counts):
        base = normal[-settings.baseline_nights:]
        mean = statistics.fmean(base) if base else None
        std = statistics.pstdev(base) if len(base) > 1 else None
        z = (counts[night] - mean) / max(std, 1.0) if mean is not None and std is not None else None
        flagged = bool(z is not None and len(base) >= 3 and z > settings.restless_z)
        if not flagged:
            normal.append(counts[night])
        out.append({"night": night.isoformat(), "restless_minutes": counts[night],
                    "baseline_mean": round(mean, 1) if mean is not None else None,
                    "baseline_std": round(std, 1) if std is not None else None,
                    "z_score": round(z, 2) if z is not None else None,
                    "flagged": flagged})
    return out[-nights:]


def check_restlessness(db: Session, household_id: str) -> Alert | None:
    """Non-urgent SMS when the last RESTLESS_NIGHTS nights are all flagged (once per streak)."""
    series = restlessness(db, household_id)
    streak = series[-settings.restless_nights:]
    if len(streak) < settings.restless_nights or not all(n["flagged"] for n in streak):
        return None
    latest = streak[-1]["night"]
    for existing in db.scalars(select(Alert).where(Alert.household_id == household_id, Alert.type == "restlessness")):
        if existing.details.get("night") == latest:
            return None
    alert = Alert(household_id=household_id, type="restlessness", severity="non_urgent",
                  details={"night": latest, "z_scores": [n["z_score"] for n in streak]})
    db.add(alert)
    db.flush()
    notify(db, alert, "caregiver", "restlessness", nights=settings.restless_nights)
    db.commit()
    return alert


def bathroom_summary(db: Session, household_id: str, days: int = 7) -> dict:
    since = utcnow() - timedelta(days=days)
    visits = list(db.scalars(select(BathroomVisit).where(BathroomVisit.household_id == household_id,
                                                          BathroomVisit.enter_ts >= since)
                             .order_by(BathroomVisit.enter_ts)))
    now = utcnow()
    per_day: dict[str, dict] = {}
    items = []
    for v in visits:
        minutes = ((v.exit_ts or now) - v.enter_ts).total_seconds() / 60
        day = local(v.enter_ts).date().isoformat()
        d = per_day.setdefault(day, {"date": day, "visits": 0, "night_visits": 0, "total_minutes": 0.0})
        d["visits"] += 1
        d["night_visits"] += _night_of(v.enter_ts) is not None
        d["total_minutes"] = round(d["total_minutes"] + minutes, 1)
        items.append({"id": v.id, "enter": v.enter_ts.isoformat(), "exit": v.exit_ts.isoformat() if v.exit_ts else None,
                      "minutes": round(minutes, 1), "in_progress": v.exit_ts is None,
                      "over_threshold": minutes > settings.bathroom_dwell_minutes})
    return {"days": list(per_day.values()), "visits": items[::-1],
            "dwell_threshold_minutes": settings.bathroom_dwell_minutes}
