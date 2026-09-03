$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$officialSource = Join-Path $workspaceRoot "COUP-table-web-interface\buildings_all.geojson"
$exportOutput = Join-Path $PSScriptRoot "physical-building-catalog-coordinates.geojson"

python "$PSScriptRoot\build.py" --export-coordinates $officialSource $exportOutput
