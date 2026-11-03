Everything this app stores can be read from your own Python scripts with `fusion_ui.api`, by shot and
pixel. It only reads: nothing is computed, and the database is opened read-only.

Run a script on the group server with the app's Python, so that it finds the app's database, cache and
data files:

```bash
/media/fusion/fusion-files/jlo044/fusion_ui/.venv/bin/python my_script.py
```

```python
from fusion_ui import api

api.shots()                                    # shots with blob parameters stored
ds = api.blob_parameters(1160616009)           # xarray, every pixel on (y, x)
ds = api.blob_parameters(1160616009, pixels=[(6, 6), (5, 6)])   # pixels as (x, y)
ds = api.velocity_fields(1160616009)           # every method's velocity field (the Fields page)
ds = api.velocity_tde(1160616009)              # the PPCF 2026 paper's TDE velocities
df = api.table()                               # pandas, one row per stored shot and live pixel
```

- Pixels are `(x, y)`: x the R index, y the Z index, as everywhere in the app.
- Every blob parameter and velocity carries its `long_name` and `units`; `R` and `Z` are in metres;
  `dead` marks the pixels the dead-pixel mask removed.
- `settings=` reads a parameter set other than the default, given as the fields that differ, e.g.
  `settings={"averages": {"window": 30}}`.
- A shot that is not stored raises `api.NotStored`, whose message is the `fusion-ui precompute` command
  that fills it.
