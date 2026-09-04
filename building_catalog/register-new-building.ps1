$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot

# build.py, calibration_markers.json gibi runtime dosyalarini proje kokune
# gore ariyor. Script nereden cagrilirsa cagrilsin ayni dizini kullan.
Push-Location -LiteralPath $projectRoot
try {
    python (Join-Path $PSScriptRoot "build.py")
}
finally {
    Pop-Location
}
