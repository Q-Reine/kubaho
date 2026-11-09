"""Kubaho event service (FastAPI). Swagger UI: http://localhost:8000/docs"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import inference, services
from .config import settings
from .db import get_db, init_db
from .models import Alert, Household, Node, Person, SmsLog
from .schemas import (AckIn, AlertOut, EventIn, EventOut, HouseholdIn, HouseholdOut, HouseholdSummary, NightOut,
                      PredictRequest, PredictResponse, SmsOut)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Kubaho API",
    version="0.1.0",
    description=(
        "Camera-free WiFi-sensing fall and wellness monitoring for older adults in rural Rwandan homes.\n\n"
        "**Flow:** ESP32-S3 node (RuView + Kubaho LoRA adapter) -> MQTT -> this service -> SMS to family "
        "-> escalation to the Community Health Worker after 5 minutes without acknowledgement.\n\n"
        "Try it: `POST /demo/seed`, then `GET /demo/sample-window?activity=fall` and paste the result into `POST /predict`."
    ),
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


def _household(db: Session, household_id: str) -> Household:
    hh = db.get(Household, household_id)
    if hh is None:
        raise HTTPException(404, f"household {household_id} not found")
    return hh


@app.get("/health", tags=["System"])
def health(db: Session = Depends(get_db)):
    try:
        card = inference.model_card()
        model = {"loaded": True, "version": card["version"], "threshold": card["threshold"]}
    except inference.ModelNotReady as exc:
        model = {"loaded": False, "error": str(exc)}
    return {"status": "ok", "database": db.bind.dialect.name, "mqtt": bool(settings.mqtt_host),
            "sms_mode": "africastalking" if settings.at_api_key else "console", "model": model}


@app.get("/model", tags=["Inference"])
def model_info():
    """Model card: classes, calibrated threshold, evaluation metrics, adapter size."""
    try:
        return inference.model_card()
    except inference.ModelNotReady as exc:
        raise HTTPException(503, str(exc))


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """Classify a 2 s CSI window with the INT8 RuView encoder + Kubaho LoRA adapter.

    Runs the same preprocessing as training (packet-loss repair, Hampel filter, 0.2-20 Hz band-pass,
    decimation to 50 Hz). On the real system this runs on the ESP32-S3 node; the endpoint exists for
    testing and for nodes that offload inference.
    """
    x = np.asarray(req.csi_amplitude, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 52 or x.shape[0] != 2 * req.fs:
        raise HTTPException(422, f"expected {2 * req.fs} x 52 amplitude matrix, got {list(x.shape)}")
    try:
        return inference.predict(x, fs=req.fs)
    except inference.ModelNotReady as exc:
        raise HTTPException(503, str(exc))


@app.post("/events", response_model=EventOut, tags=["Ingest"])
def ingest(event: EventIn, db: Session = Depends(get_db)):
    """HTTP ingest for node events (same handler as the MQTT topic `kubaho/<node_id>/events`)."""
    try:
        stored, alert = services.ingest_event(db, event.model_dump())
    except services.IngestError as exc:
        raise HTTPException(422, str(exc))
    if stored is None:
        return EventOut(accepted=True, duplicate=True)
    return EventOut(accepted=True, event_id=stored.id, alert_id=alert.id if alert else None)


@app.get("/households", response_model=list[HouseholdSummary], tags=["Households"])
def list_households(db: Session = Depends(get_db)):
    out = []
    for hh in db.scalars(select(Household).order_by(Household.id)):
        elder = next((p.name for p in hh.people if p.role == "elder"), None)
        open_alerts = db.scalar(select(func.count(Alert.id)).where(
            Alert.household_id == hh.id, Alert.status.in_(["open", "escalated"]), Alert.severity != "info"))
        last = db.scalar(select(func.max(Alert.created_at)).where(Alert.household_id == hh.id))
        out.append(HouseholdSummary(id=hh.id, name=hh.name, village=hh.village, elder=elder, open_alerts=open_alerts,
                                    nodes_online=sum(n.status == "online" for n in hh.nodes),
                                    nodes_total=len(hh.nodes), last_alert_at=last))
    return out


@app.post("/households", response_model=HouseholdOut, status_code=201, tags=["Households"])
def create_household(body: HouseholdIn, db: Session = Depends(get_db)):
    if db.get(Household, body.id):
        raise HTTPException(409, f"household {body.id} already exists")
    hh = Household(id=body.id, name=body.name, cell=body.cell, village=body.village)
    hh.people = [Person(**p.model_dump()) for p in body.people]
    hh.nodes = [Node(**n.model_dump()) for n in body.nodes]
    db.add(hh)
    db.commit()
    return hh


@app.get("/households/{household_id}", response_model=HouseholdOut, tags=["Households"])
def get_household(household_id: str, db: Session = Depends(get_db)):
    return _household(db, household_id)


@app.get("/households/{household_id}/alerts", response_model=list[AlertOut], tags=["Alerts"])
def household_alerts(household_id: str, limit: int = Query(50, le=500), db: Session = Depends(get_db)):
    _household(db, household_id)
    return list(db.scalars(select(Alert).where(Alert.household_id == household_id)
                           .order_by(Alert.created_at.desc()).limit(limit)))


@app.post("/alerts/{alert_id}/ack", response_model=AlertOut, tags=["Alerts"])
def ack_alert(alert_id: int, body: AckIn, db: Session = Depends(get_db)):
    """Acknowledge an alert. The outcome (real fall / false alarm) is field ground truth for the pilot."""
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, f"alert {alert_id} not found")
    return services.acknowledge(db, alert, body.outcome, body.by)


@app.get("/households/{household_id}/restlessness", response_model=list[NightOut], tags=["Wellness"])
def household_restlessness(household_id: str, nights: int = Query(14, ge=1, le=90), db: Session = Depends(get_db)):
    """Minutes of in-bed motion between 22:00 and 06:00 per night, with a z-score against the previous
    7 nights. Three flagged nights in a row (z > 2) trigger a non-urgent SMS."""
    _household(db, household_id)
    return services.restlessness(db, household_id, nights)


@app.get("/households/{household_id}/bathroom", tags=["Wellness"])
def household_bathroom(household_id: str, days: int = Query(7, ge=1, le=60), db: Session = Depends(get_db)):
    """Bathroom visits from RuView zone presence; a visit over 30 minutes raises an urgent alert."""
    _household(db, household_id)
    return services.bathroom_summary(db, household_id, days)


@app.get("/households/{household_id}/sms", response_model=list[SmsOut], tags=["SMS"])
def household_sms(household_id: str, limit: int = Query(50, le=500), db: Session = Depends(get_db)):
    return list(db.scalars(select(SmsLog).where(SmsLog.household_id == household_id)
                           .order_by(SmsLog.created_at.desc()).limit(limit)))


@app.post("/sms/incoming", tags=["SMS"])
def sms_incoming(from_: str = Form(..., alias="from"), text: str = Form(...), db: Session = Depends(get_db)):
    """Africa's Talking incoming-SMS callback. A reply of `1` confirms the latest urgent alert, `2` marks it
    a false alarm."""
    alert = services.handle_incoming_sms(db, from_, text)
    return {"matched_alert": alert.id if alert else None}


@app.post("/demo/seed", tags=["Demo"])
def demo_seed(reset: bool = False, db: Session = Depends(get_db)):
    """Create three demo households in Bumbogo with 14 nights of history (fictional people)."""
    from .seed import seed

    return seed(db, reset=reset)


@app.get("/demo/sample-window", tags=["Demo"])
def demo_sample_window(activity: str = Query("fall_forward", description="e.g. fall_forward, sit_on_mat, "
                                             "kneel_pray, squat_stool, walk, bend_sweep, restless_bed, empty"),
                       seed: int | None = None):
    """A synthetic raw CSI amplitude window (200 x 52) to paste into POST /predict."""
    from kubaho.synthetic import KUBAHO_ACTIVITIES, make_layout, make_subject, simulate_clip

    if activity not in KUBAHO_ACTIVITIES:
        raise HTTPException(422, f"activity must be one of {sorted(KUBAHO_ACTIVITIES)}")
    rng = np.random.default_rng(seed if seed is not None else uuid.uuid4().int % 2**32)
    csi = simulate_clip(activity, make_layout("pilot", 999, rural=True), make_subject(rng), rng)
    window = np.nan_to_num(np.abs(csi[50:250]))
    return {"activity": activity, "expected_class": KUBAHO_ACTIVITIES[activity], "fs": 100,
            "csi_amplitude": np.round(window, 4).tolist()}
