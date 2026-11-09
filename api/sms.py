"""SMS templates (Kinyarwanda / English) and delivery through Africa's Talking."""

from __future__ import annotations

import logging

import httpx

from .config import settings

log = logging.getLogger("kubaho.sms")

TEMPLATES = {
    "fall": {
        "en": "KUBAHO ALERT: possible fall detected for {elder} at {time}. Please check on them now. "
              "Reply 1 if it was a real fall, 2 if false alarm.",
        "rw": "KUBAHO: {elder} ashobora kuba yaguye saa {time}. Nyamuneka mumugenzure ubu. "
              "Subiza 1 niba yaguye koko, 2 niba atari byo.",
    },
    "escalation": {
        "en": "KUBAHO ESCALATION: possible fall for {elder} ({village}) at {time} was not acknowledged by "
              "family within {minutes} min. Please visit. Reply 1 real fall, 2 false alarm.",
        "rw": "KUBAHO: {elder} ({village}) ashobora kuba yaguye saa {time}, umuryango ntiwabyemeje mu "
              "minota {minutes}. Nyamuneka mumusure. Subiza 1 niba yaguye koko, 2 niba atari byo.",
    },
    "bathroom_dwell": {
        "en": "KUBAHO URGENT: {elder} has been in the bathroom for over {minutes} min. Please check on them. "
              "Reply 1 if help was needed, 2 if all is well.",
        "rw": "KUBAHO: {elder} amaze iminota irenga {minutes} mu bwiherero. Nyamuneka mumugenzure. "
              "Subiza 1 niba yari akeneye ubufasha, 2 niba ari amahoro.",
    },
    "restlessness": {
        "en": "KUBAHO: {elder} has had unusually restless nights for {nights} nights in a row. "
              "Not an emergency - consider a visit or a CHW check-up.",
        "rw": "KUBAHO: {elder} amaze amajoro {nights} asinzira nabi kurusha uko bisanzwe. "
              "Si ibyihutirwa - mwamusura cyangwa mukabimenyesha umujyanama w'ubuzima.",
    },
}


def render(kind: str, language: str, **values) -> str:
    template = TEMPLATES[kind].get(language) or TEMPLATES[kind]["en"]
    return template.format(**values)


def send_sms(phone: str, message: str) -> str:
    """Send one SMS. Returns a status string: sent | logged | failed."""
    if not settings.at_api_key:
        log.info("[sms:console] to=%s %s", phone, message)
        return "logged"
    host = "api.sandbox.africastalking.com" if settings.at_username == "sandbox" else "api.africastalking.com"
    data = {"username": settings.at_username, "to": phone, "message": message}
    if settings.at_sender_id:
        data["from"] = settings.at_sender_id
    try:
        resp = httpx.post(
            f"https://{host}/version1/messaging",
            data=data,
            headers={"apiKey": settings.at_api_key, "Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        return "sent"
    except httpx.HTTPError as exc:
        log.error("[sms] delivery to %s failed: %s", phone, exc)
        return "failed"
