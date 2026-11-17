param(
    [ValidateSet('play','train','evaluate','test')]
    [string]$Mode = 'play'
)
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Push-Location $PSScriptRoot
try {
    if ($Mode -eq 'test') { & $projectPython -m pytest -q }
    else { & $projectPython -m flappy_ai $Mode }
} finally {
    Pop-Location
}
