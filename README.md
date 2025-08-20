## 🎱 Pool Shot Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Enabled-orange?logo=numpy&logoColor=white)
![OS](https://img.shields.io/badge/OS-Windows%2010+-blue?logo=windows&logoColor=white)

Smart, real‑time pool shot prediction with a clean side‑by‑side view: raw video on the left and prediction overlay on the right — all in a single window.

---

### ✨ Features

- **Single-window UI**: Left = live video, Right = prediction overlay
- **Automatic table detection**: Finds table edges and corners
- **Ball detection**: Detects balls via contours and template matching
- **Cue analysis**: Extracts cue line and computes collision points
- **Reflection & pocket prediction**: Simulates bounces and predicts IN/OUT
- **Video export**: Saves processed output to `outputvideo.mp4`

---

### Output

<img width="1920" height="1080" alt="Screenshot (134)" src="https://github.com/user-attachments/assets/0c850dec-0224-4866-8c93-79faab0f6703" />

<img width="1920" height="1080" alt="Screenshot (135)" src="https://github.com/user-attachments/assets/5835ccf7-5a35-441b-b748-874796cbdfa5" />

---

### 🚀 Quickstart (Windows)

1) Clone or open the project folder in your IDE.

2) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies:

```powershell
pip install -r requirements.txt
```

4) Put your input video at the project root as `video.mp4`.

5) Ensure ball templates exist:
- `Resources/Images/pic1.png`
- `Resources/Images/pic3.png`

6) Run it:

```powershell
python pool_shot_prediction.py
```

Press `q` to quit.

---

### 🧠 How it works

The core pipeline lives in `pool_shot_prediction.py` and follows these stages:

- Convert frame to HSV, segment the table felt
- Detect table boundary lines and compute corners
- Build ROI masks to isolate valid play area and pockets
- Detect balls using contours; optionally use template matching for robustness
- Determine cue direction and compute the collision point
- Predict target ball trajectory, reflections, and pocket hit
- Render overlays onto a dedicated prediction frame (`pred_frame`)
- Display side‑by‑side: raw `original_frame` and `pred_frame`

Geometry utilities are implemented in `instersectioncheck.py`:
- `findintersection(...)`: Finds next rail intersection and normal
- `insidepocket(...)`: Checks if a point lies within a pocket radius

---

### 🛠️ Configuration

Edit these in `pool_shot_prediction.py` to tune behavior:

- `resize_width`: Horizontal size per view (final window is roughly 2x this)
- `timelimits`: Number of frames to keep the prediction visible (timeout)
- Green felt mask HSV bounds: `np.array([56, 161, 38])` to `np.array([71, 255, 94])`
- White cue threshold for Hough lines and ball radius range `22–35`
- Pocket detection radius in `insidepocket(...)` (default ≈ 40 px)

---

### 🧩 Project structure

```
Pool Shot Predictor/
├─ pool_shot_prediction.py      # Main pipeline and UI
├─ instersectioncheck.py        # Geometry helpers (intersection, pockets)
├─ Resources/
│  └─ Images/
│     ├─ pic1.png               # Ball template 1
│     └─ pic3.png               # Ball template 2
├─ video.mp4                    # Your input video (add this)
├─ outputvideo.mp4              # Saved results (generated)
└─ README.md                    # This file
```

---

### ❗ Troubleshooting

- **Black/blank preview**: Check that `video.mp4` exists and is readable.
- **Left side not playing during timeout**: We refresh the GUI each frame; if it still feels slow, reduce values in `timelimits`.
- **Wrong table/ball colors**: Adjust HSV thresholds for your lighting.
- **Prediction seems off**: Verify ball radius thresholds and Hough line params.
- **Window too large**: Lower `resize_width`.

---

### 🙌 Acknowledgements

Built with OpenCV, NumPy, and lots of geometry love. Enjoy your shots! 🎯


## Thank You | MkM😎
### Developed By Manikandan M 
