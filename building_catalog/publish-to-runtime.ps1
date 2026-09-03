$ErrorActionPreference = "Stop"

$workingCatalog = Join-Path $PSScriptRoot "physical-building-catalog.json"
$runtimeCatalog = Join-Path (Split-Path -Parent $PSScriptRoot) "physical-building-catalog.json"

Copy-Item -LiteralPath $workingCatalog -Destination $runtimeCatalog -Confirm
