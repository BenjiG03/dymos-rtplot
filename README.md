# dymos-rtplot

`dymos-rtplot` is a local fork of OpenMDAO `rtplot` aimed at Dymos debugging, including
live trajectory views, optimizer diagnostics, and total-Jacobian inspection tabs.

## Development install

From the repository root:

```powershell
conda activate optview_tutorial
python -m pip install -e .
```

## Run the scratch mission

```powershell
python -m dymos_rtplot.rtplot scratch\multiphase_dymos_rtplot_mission.py
```

Or use the console script after the editable install:

```powershell
dymos-rtplot scratch\multiphase_dymos_rtplot_mission.py
```

You can also run the installed package directly:

```powershell
python -m dymos_rtplot scratch\multiphase_dymos_rtplot_mission.py
```

By default the realtime server now prints a `127.0.0.1` URL instead of trying to
launch the system browser automatically. This avoids browser-launch policy issues
on managed Windows machines. If you want the old behavior, add `--open-browser`.
