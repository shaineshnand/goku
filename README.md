# Dragon Ball Kamehameha (Hand Gesture)

Create a webcam effect where bringing both hands together charges an energy ball, and separating hands shoots a Kamehameha beam.

## Features

- Detects up to 2 hands using MediaPipe Hand Landmarker
- Charges energy when hands are close together
- Shoots beam when hands spread apart quickly
- Shows HUD text similar to your reference image

## Requirements

- Python 3.10+ (recommended)
- Webcam
- Windows/macOS/Linux

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python .\kamehameha.py
```

Press `q` to quit.

## Controls / Gesture

1. Keep both hands visible to the camera.
2. Move hands close together to charge.
3. Spread hands apart to fire.

## Notes

- If webcam cannot open, close other apps using camera and allow camera permissions.
- For better detection, use clear lighting and keep hands fully visible.
- On first run, it downloads `hand_landmarker.task` automatically.
- If tracking is shaky, use brighter lighting and keep hands clearly in frame.
- Optional custom sound files:
  - `assets/sfx/charge.wav`
  - `assets/sfx/fire.wav`
  - If missing, the app falls back to Windows beep sounds.