# Physical Building Catalog Workspace

This folder is the working area for mapping physical table objects to official
building footprints. It is deliberately separate from the catalog used by the
running server, so incomplete edits cannot accidentally affect a live session.

## Files

- `physical-building-catalog.json`: editable working catalog with marker IDs and
  table-local building geometries.
- `physical-building-catalog-coordinates.geojson`: QGIS-readable validation
  output. Its main geometry is the official WGS84 longitude/latitude footprint;
  `properties.local_geometry` contains the corresponding table-local geometry.
- `build.py`: interactive catalog builder and coordinate export tool.
- `register-new-building.ps1`, `export-for-qgis.ps1`, and
  `publish-to-runtime.ps1`: convenience commands described below.

The server continues to read `../physical-building-catalog.json`. Nothing in
this folder changes that runtime file unless you explicitly publish it.

## Register a new physical building

Place only the building being registered on the table, then run from the
repository root:

```powershell
.\building_catalog\register-new-building.ps1
```

Enter its `building_id` (for example `G17`). The tool observes its stable ArUco
marker, matches the ID against `../COUP-table-web-interface/buildings_all.geojson`,
and updates this folder's working catalog. Enter `done` when finished.

Equivalent direct command:

```powershell
python .\building_catalog\build.py
```

## Validate official coordinates in QGIS

```powershell
.\building_catalog\export-for-qgis.ps1
```

Open `physical-building-catalog-coordinates.geojson` in QGIS. Only buildings in
the working catalog are exported. Matching is case-insensitive by `building_id`,
and the command fails clearly if an ID is absent from the official source.

## Publish the finished catalog to the runtime

Review the working JSON and QGIS output first. Then run:

```powershell
.\building_catalog\publish-to-runtime.ps1
```

PowerShell asks for confirmation before replacing `../physical-building-catalog.json`.
This explicit copy is the only publishing step; the server keeps reading the
root file as before.
