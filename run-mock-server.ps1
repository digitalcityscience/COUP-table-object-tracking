<#
.SYNOPSIS
    Run the tracking server against a simulated table -- no cameras, no rig, no projector.

.DESCRIPTION
    The counterpart to run-server.ps1 for frontend development away from the tangible table.
    It runs the real server.py; only the camera detection thread is replaced (see mock_server.py).
    The websocket speaks the same protocol on the same port, so TOSCA-2 needs no changes to talk
    to it.

    An interactive prompt lets you move, turn and remove the physical blocks while the frontend
    watches. Type '?' at that prompt for the command list.

.EXAMPLE
    .\run-mock-server.ps1
    Blocks sit still (with detection noise), waiting for you to move them from the prompt.

.EXAMPLE
    .\run-mock-server.ps1 -Motion drift
    Blocks wander and spin on their own -- for watching the frontend handle continuous movement.

.EXAMPLE
    .\run-mock-server.ps1 -Reset
    Throw away mock_state/ first, so registrations from an earlier session are gone.
#>
[CmdletBinding()]
param(
    # still: frozen and exactly repeatable. jitter: detection noise only. drift: blocks move.
    [ValidateSet("still", "jitter", "drift")]
    [string]$Motion = "jitter",

    [int]$Port = 8053,

    # Throw away the sandboxed catalogs in mock_state/ and start from a fresh copy of the real ones.
    [switch]$Reset,

    # Write registrations to the REAL catalogs instead of the sandbox. Rarely what you want.
    [switch]$NoSandbox,

    # Echo every snapshot the server sends. Very noisy -- five GeoJSON dumps a second.
    [switch]$Verbose_Feed
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install it from https://docs.astral.sh/uv/getting-started/installation/"
}

Push-Location $ProjectDir
try {
    $ServerArgs = @("mock_server.py", "--motion", $Motion, "--port", $Port)
    if ($Reset) { $ServerArgs += "--reset" }
    if ($NoSandbox) { $ServerArgs += "--no-sandbox" }
    if ($Verbose_Feed) { $ServerArgs += "--verbose" }
    uv run --python 3.13 --with-requirements requirements.txt -- python @ServerArgs
}
finally {
    Pop-Location
}
