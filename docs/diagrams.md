# Kubaho design diagrams

All diagrams are Mermaid, so GitHub renders them directly.

## System architecture (proposal Figure 3)

```mermaid
flowchart LR
    subgraph Home["Rural home (Bumbogo)"]
        R["WiFi / 4G MiFi router"] -. CSI .-> N["ESP32-S3 node<br/>RuView firmware<br/>INT8 encoder + Kubaho LoRA"]
        N2["Bathroom node<br/>(zone presence)"]
    end
    N -- "MQTT + TLS, QoS 1<br/>kubaho/&lt;node&gt;/events" --> B[("Eclipse Mosquitto")]
    N2 -- MQTT --> B
    B --> W["Alert worker<br/>ingest, escalation,<br/>bathroom dwell, restlessness"]
    W --> DB[("PostgreSQL + TimescaleDB")]
    API["FastAPI event service<br/>/events /predict /alerts ..."] --> DB
    W -- "Africa's Talking API" --> SMS["SMS (Kinyarwanda / English)"]
    SMS --> F["Family caregiver"]
    SMS -- "no ack in 5 min" --> C["Community Health Worker"]
    F -- "reply 1 / 2" --> API
    D["React PWA dashboard"] --> API
    REG["Model registry<br/>(adapters)"] -. "OTA adapter push" .-> N
```

## ML training pipeline (proposal Figure 4)

```mermaid
flowchart LR
    A["Raw CSI<br/>100 packets/s, 52 subcarriers"] --> Q["Quality phase<br/>loss repair, phase sanitisation,<br/>Hampel, band-pass 0.2-20 Hz,<br/>drop sessions &gt;10% loss"]
    Q --> S["Subject- and layout-<br/>disjoint splits"]
    S --> W["2 s windows<br/>0.25 s stride, 50 Hz"]
    W --> E["Frozen pretrained<br/>CSI encoder (128-d)"]
    W --> F["Statistical features<br/>(classical baselines)"]
    E --> L["LoRA r=4 on q, v<br/>+ 8-class head<br/>focal loss, AdamW"]
    L --> V["Leave-one-layout-out<br/>threshold calibration"]
    V --> X["Merge, ONNX, INT8<br/>model card, registry"]
    X --> O["OTA to ESP32-S3"]
```

## Sequence: fall detection, alerting and escalation (proposal Figure 8)

```mermaid
sequenceDiagram
    autonumber
    participant N as ESP32-S3 node
    participant M as MQTT broker
    participant W as Alert worker
    participant DB as Database
    participant AT as Africa's Talking
    participant F as Family caregiver
    participant C as CHW
    N->>N: CSI window -> INT8 model, P(fall) >= threshold
    N->>M: publish fall event (QoS 1)
    M->>W: deliver event
    W->>DB: validate, de-duplicate, store event + alert
    W->>AT: send SMS (caregiver's language)
    AT->>F: "Possible fall... reply 1 / 2"
    alt caregiver replies within 5 min
        F->>AT: "1" (real fall) or "2" (false alarm)
        AT->>W: POST /sms/incoming
        W->>DB: alert acknowledged (ground truth)
    else no acknowledgement after 5 min
        W->>DB: status = escalated
        W->>AT: escalation SMS
        AT->>C: "Not acknowledged by family... please visit"
        C->>AT: reply 1 / 2
        AT->>W: POST /sms/incoming
    end
```

## Entity relationship diagram (proposal Figure 7)

```mermaid
erDiagram
    HOUSEHOLD ||--o{ PERSON : has
    HOUSEHOLD ||--o{ NODE : has
    HOUSEHOLD ||--o{ EVENT : records
    HOUSEHOLD ||--o{ ALERT : raises
    HOUSEHOLD ||--o{ MOTION_MINUTE : aggregates
    HOUSEHOLD ||--o{ BATHROOM_VISIT : logs
    NODE ||--o{ EVENT : publishes
    ALERT ||--o{ SMS_LOG : triggers
    HOUSEHOLD { string id PK  string name  string sector  string cell  string village }
    PERSON { int id PK  string household_id FK  string role  string name  string phone  string language }
    NODE { string id PK  string household_id FK  string zone  string status  datetime last_seen  string adapter_version }
    EVENT { int id PK  string event_uid UK  string node_id FK  string type  float confidence  datetime ts }
    ALERT { int id PK  string type  string severity  string status  datetime created_at  datetime escalated_at  string ack_outcome }
    MOTION_MINUTE { string household_id PK  datetime ts PK  float energy }
    BATHROOM_VISIT { int id PK  datetime enter_ts  datetime exit_ts  bool alerted }
    SMS_LOG { int id PK  int alert_id FK  string direction  string phone  string message  string status }
    ADAPTER { string version PK  int rank  float fall_f1  float threshold  string path }
```

## Use cases (proposal Figure 5)

```mermaid
flowchart LR
    E(("Elder")) --- U1["Is monitored passively<br/>(no wearable, no camera)"]
    F(("Family caregiver")) --- U2["Receive fall / bathroom SMS"]
    F --- U3["Acknowledge alert<br/>(SMS reply or dashboard)"]
    F --- U4["View nightly restlessness<br/>and bathroom trends"]
    C(("CHW")) --- U5["Receive escalations"]
    C --- U3
    C --- U4
    R(("Researcher / admin")) --- U6["Register household and nodes"]
    R --- U7["Train and push LoRA adapter (OTA)"]
```
