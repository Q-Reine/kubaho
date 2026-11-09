"""ORM tables (matches the ERD in docs/diagrams.md)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, utcnow


class Household(Base):
    __tablename__ = "households"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    sector: Mapped[str] = mapped_column(String(60), default="Bumbogo")
    cell: Mapped[str] = mapped_column(String(60), default="")
    village: Mapped[str] = mapped_column(String(60), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    people: Mapped[list["Person"]] = relationship(back_populates="household", cascade="all, delete-orphan")
    nodes: Mapped[list["Node"]] = relationship(back_populates="household", cascade="all, delete-orphan")


class Person(Base):
    """Elder, family caregiver or community health worker (CHW)."""

    __tablename__ = "people"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    household_id: Mapped[str] = mapped_column(ForeignKey("households.id"))
    role: Mapped[str] = mapped_column(String(16))  # elder | caregiver | chw
    name: Mapped[str] = mapped_column(String(120))
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    language: Mapped[str] = mapped_column(String(2), default="rw")  # rw | en

    household: Mapped[Household] = relationship(back_populates="people")


class Node(Base):
    """An ESP32-S3 RuView sensing node."""

    __tablename__ = "nodes"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    household_id: Mapped[str] = mapped_column(ForeignKey("households.id"))
    zone: Mapped[str] = mapped_column(String(32), default="living_room")  # living_room | bathroom
    status: Mapped[str] = mapped_column(String(16), default="unknown")   # online | offline | unknown
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    adapter_version: Mapped[str] = mapped_column(String(32), default="kubaho-lora-r4-v0.1")

    household: Mapped[Household] = relationship(back_populates="nodes")


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_uid: Mapped[str] = mapped_column(String(64), unique=True)  # de-duplicates MQTT QoS 1 redelivery
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id"))
    household_id: Mapped[str] = mapped_column(ForeignKey("households.id"), index=True)
    type: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


class MotionMinute(Base):
    """Per-minute in-room motion energy, used for the night-time restlessness index."""

    __tablename__ = "motion_minutes"
    household_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    energy: Mapped[float] = mapped_column(Float)


class BathroomVisit(Base):
    __tablename__ = "bathroom_visits"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    household_id: Mapped[str] = mapped_column(ForeignKey("households.id"), index=True)
    enter_ts: Mapped[datetime] = mapped_column(DateTime)
    exit_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    alerted: Mapped[bool] = mapped_column(Boolean, default=False)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    household_id: Mapped[str] = mapped_column(ForeignKey("households.id"), index=True)
    type: Mapped[str] = mapped_column(String(32))       # fall | bathroom_dwell | restlessness | node_offline
    severity: Mapped[str] = mapped_column(String(16))   # urgent | non_urgent | info
    status: Mapped[str] = mapped_column(String(16), default="open")  # open | escalated | acknowledged
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    escalated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ack_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ack_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    ack_outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)  # confirmed | false_alarm
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class SmsLog(Base):
    __tablename__ = "sms_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_id: Mapped[int | None] = mapped_column(ForeignKey("alerts.id"), nullable=True)
    household_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    direction: Mapped[str] = mapped_column(String(8), default="out")  # out | in
    phone: Mapped[str] = mapped_column(String(20))
    recipient_role: Mapped[str | None] = mapped_column(String(16), nullable=True)
    message: Mapped[str] = mapped_column(String(480))
    status: Mapped[str] = mapped_column(String(32))  # sent | logged | failed | received
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Adapter(Base):
    """Model registry entry for a LoRA adapter pushed over the air."""

    __tablename__ = "adapters"
    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer)
    fall_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float] = mapped_column(Float)
    path: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
