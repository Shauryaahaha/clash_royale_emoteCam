# Clash Emote Vision

I built this project using Python for real time processing, OpenCV for webcam capture and visual overlays, and MediaPipe (Hands and Face Mesh) for hand and facial landmark detection. NumPy was used for numerical computations and motion analysis, while Pillow (PIL) handled animated GIF decoding. The system incorporates temporal smoothing, rule based gesture recognition using geometric and velocity based heuristics, and a low latency real time pipeline with robust camera handling and environment based configuration.

## Fallback
If MediaPipe cannot be installed on your machine (common with some Python versions or 32-bit installs), either:
- Install 64-bit Python 3.10 and recreate venv (recommended), **or**

- Use the OpenCV-only fallback (see README for instructions).

- Feel free to play around with it :)

