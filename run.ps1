param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ProjectArguments
)
$ErrorActionPreference = 'Stop'
$projectPython = if ($env:VTLAB_PYTHON) {
    $env:VTLAB_PYTHON
} elseif (Test-Path -LiteralPath 'D:\Python\python.exe') {
    'D:\Python\python.exe'
} else {
    (Get-Command python -ErrorAction Stop).Source
}
Push-Location $PSScriptRoot
try {
    & $projectPython (Join-Path $PSScriptRoot 'run.py') @ProjectArguments
    $projectExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $projectExitCode
