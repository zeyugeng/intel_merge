# Archive Notes

Legacy code is archived in place to avoid breaking local hardware experiments:

- `intelcup/` is an earlier prototype and now has a local README explaining replacements.
- `pantilt_control.py` is a standalone serial-servo lab script. Its reusable backend has been integrated as `PythonProject/core/serial_ptz.py`.

The active runtime path is `PythonProject/scripts/run_sound_ptz_all.py`.
