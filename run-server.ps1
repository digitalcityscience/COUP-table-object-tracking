[CmdletBinding()]
param(
    [ValidateSet("web", "unity")]
    [string]$Client = "web"
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install it from https://docs.astral.sh/uv/getting-started/installation/"
}

Push-Location $ProjectDir
try {
    uv run --python 3.13 --with-requirements requirements.txt -- python server.py --client $Client
}
finally {
    Pop-Location
}
