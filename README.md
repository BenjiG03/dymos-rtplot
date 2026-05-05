# dymos-rtplot

`dymos-rtplot` is a local fork of OpenMDAO `rtplot` aimed at Dymos debugging, including
live trajectory views, optimizer diagnostics, and total-Jacobian inspection tabs.

## Development install

From the repository root:

```powershell
python -m pip install -e .
```

## Run with a dymos problem

```powershell
dymos-rtplot dymos_problem.py
```

To avoid being blocked by windows firewall on some machines, the default behavior is to 
only print the urls where the dashboard tabs are hosted. To have the dashboard open automatically,
add --open-browser`.
