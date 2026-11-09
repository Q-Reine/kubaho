# Data

No raw data is committed. `data/processed/` is a cache that the notebook creates.

| Dataset | Status | How to get it |
|---|---|---|
| Kubaho-Lab (20 volunteers, mock rural room, 2 layouts) | Synthetic stand-in in `src/kubaho/synthetic.py` | Collected under RNEC approval with informed consent; stored de-identified on the lab workstation |
| Kubaho-Field (8 pilot homes × 6 weeks) | 8-household pilot | Event logs flow into the API database; caregiver/CHW acknowledgements are the ground truth |
| MM-Fi | Public CSI dataset | Request access at https://ntu-aiot-lab.github.io/mm-fi (non-commercial research licence) |
| RuView pretrained encoder | Stand-in encoder trained in the notebook | Weights from the π RuView repository (MIT) |

## Plugging in real recordings

`kubaho.data.build_kubaho_lab` returns `X` with shape `(events, 3 crops, 100, 52)` plus a metadata table
(`subject, layout, session, activity, label, class_name, rep, packet_loss`). To use real data, write a loader
with the same output. Read each labelled event's raw complex CSI (100 packets/s), cut three 2 s windows at
0 / 0.5 / 1.0 s, and pass each through `kubaho.preprocess.preprocess_window`. Then cache the result as
`data/processed/kubaho_lab.npz` + `kubaho_lab_meta.csv`. The notebook runs unchanged.

## Privacy

Raw CSI stays on the node or the lab workstation. Only events and aggregates reach the server. Households
are pseudonymised (IDs). No images or audio are recorded in pilot homes. Data handling follows Rwanda's Law
No. 058/2021 on the protection of personal data and privacy.
