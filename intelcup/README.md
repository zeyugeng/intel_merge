# Legacy Prototype Code

This directory is kept as historical reference from the earlier prototype.
The active application code lives in `PythonProject/core` and is launched from
`PythonProject/scripts`.

Current replacements:

- `MicrophoneArray.py` -> `PythonProject/core/odas_bridge.py` + `PythonProject/core/sound_client.py`
- `SoundPredict.py` -> `PythonProject/core/birdnet_infer.py`
- `Camera.py` -> `PythonProject/core/camera.py`
- `Ptz.py` -> `PythonProject/core/ptz_camera.py`
- `main.py` -> `PythonProject/scripts/run_sound_client.py`

Do not use these files for new runtime flows.
