# Clash Emote Vision - Prototype

## Setup (Windows)
1. Create venv:
python -m venv venv

2. Activate (PowerShell):
.\venv\Scripts\Activate.ps1

3. Upgrade pip and install:
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

*If `pip install mediapipe` fails, see README fallback section.*

4. Ensure `.env` exists (we provided one). Put emote PNGs in `src/assets/emotes/`.

5. Run:
python check_env.py
python demo.py

## Fallback
If MediaPipe cannot be installed on your machine (common with some Python versions or 32-bit installs), either:
- Install 64-bit Python 3.10 and recreate venv (recommended), **or**
- Use the OpenCV-only fallback (see README for instructions).