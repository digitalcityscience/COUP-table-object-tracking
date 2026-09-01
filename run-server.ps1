[CmdletBinding()]
param(
    [ValidateSet("web", "unity")]
    [string]$Client = "web",
    [switch]$Calibrate
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install it from https://docs.astral.sh/uv/getting-started/installation/"
}

Push-Location $ProjectDir
try {
    $ServerArgs = @("server.py", "--client", $Client)
    if ($Calibrate) {
        $ServerArgs += "--calibrate"
    }
    uv run --python 3.13 --with-requirements requirements.txt -- python @ServerArgs
}
finally {
    Pop-Location
}
