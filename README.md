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

## Multiwindow dashboard

The optimizer dashboard can run in the default tabbed mode or as one process per
dashboard tab:

```powershell
python -m dymos_rtplot.rtplot scratch\multiphase_dymos_rtplot_mission.py --dashboard-mode multiwindow
```

By default multiwindow mode assigns predictable ports starting at `57003`, so the
tab URLs are stable across launches:

- `case-plotter`: `http://127.0.0.1:57003/`
- `trajectory`: `http://127.0.0.1:57004/`
- `series`: `http://127.0.0.1:57005/`
- `jacobian-entries`: `http://127.0.0.1:57006/`
- `jacobian-heatmap`: `http://127.0.0.1:57007/`

You can limit the launch to specific tabs:

```powershell
python -m dymos_rtplot.rtplot scratch\multiphase_dymos_rtplot_mission.py --dashboard-mode multiwindow --tabs case-plotter,trajectory,series
```

Available tab names are:

- `case-plotter`
- `trajectory`
- `series`
- `jacobian-entries`
- `jacobian-heatmap`

You can also pin launched tabs to CPU cores:

```powershell
python -m dymos_rtplot.rtplot scratch\multiphase_dymos_rtplot_mission.py --dashboard-mode multiwindow --tabs case-plotter,trajectory --tab-core case-plotter=0,trajectory=1
```

To move the stable URL block, set a different base port:

```powershell
python -m dymos_rtplot.rtplot scratch\multiphase_dymos_rtplot_mission.py --dashboard-mode multiwindow --base-port 58000
```
