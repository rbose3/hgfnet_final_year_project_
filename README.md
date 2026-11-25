# HGFNet Streamlit Application

A Streamlit interface for the original trained HGFNet implementation and METR-LA dataset.

## Project structure

```text
HGFNet_Streamlit/
├── app.py
├── model.py
├── utils.py
├── requirements.txt
├── README.md
├── checkpoints/
│   └── HGFNet.pth
└── data/
    └── METR-LA.h5
```

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Implementation notes

- The model architecture in `model.py` reproduces the notebook's module definitions and parameter names so the supplied state dictionary loads strictly.
- Input: 12 historical steps × 207 sensors × 1 feature.
- Output: 12 future steps × 207 sensors × 1 feature.
- Sampling interval: 5 minutes.
- The scaler is not stored in the `.pth` checkpoint. `utils.py` recreates it by fitting `StandardScaler` on the first 70% of the original METR-LA time series, matching the notebook.
- Missing/zero readings are retained in preprocessing as in the notebook. The app excludes zero actual values only when computing retrospective MAPE/metrics.
- The app displays live predictions and, when future observations exist in the dataset, retrospective metrics for the selected window. These local-window metrics are not the aggregate test metrics in the report.
- The notebook's `TemporalStateSpace` class is actually a gated depthwise Conv1D block; this app preserves that implementation.

## Data use

Keep the dataset in `data/METR-LA.h5` and checkpoint in `checkpoints/HGFNet.pth`. Do not rename the files unless you also update the paths in `app.py`.
