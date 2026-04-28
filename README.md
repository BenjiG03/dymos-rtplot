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
