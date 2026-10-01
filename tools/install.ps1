param(
    [string]$Target = "",
    [ValidateSet("core", "all")][string]$Profile = "all",
    [string]$Python = "python",
    [switch]$Interactive
)

$env:SENTINEL_PROFILE = $Profile
$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = (Join-Path $RepoRoot "src") + [IO.Path]::PathSeparator + $env:PYTHONPATH
if ($Interactive -or -not $Target) {
    $WizardArgs = @("-m", "sentinel_evc.install_wizard", "--source", $RepoRoot, "--profile", $Profile)
    if ($Target) { $WizardArgs += @("--target", $Target) }
    & $Python @WizardArgs
    exit $LASTEXITCODE
}
& $Python -c 'from sentinel_evc.installation import install_system; import os,sys; r=install_system(sys.argv[1], profile=os.environ["SENTINEL_PROFILE"], source=sys.argv[2], python_executable=sys.executable, build_app=False); print(r.target)' $Target $RepoRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
