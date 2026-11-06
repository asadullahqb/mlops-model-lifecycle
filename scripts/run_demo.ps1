# One-command local demo of the closed loop (Windows PowerShell).
# Runs the components directly; no Docker needed.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "==> 1/5 install dependencies"
pip install -q -r requirements.txt

Write-Host "==> 2/5 reset dataset + train baseline model"
python -m src.data reset
python -m src.train

Write-Host "==> 3/5 start serve (background) and hit /predict"
$proc = Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","src.serve:app","--host","127.0.0.1","--port","8000" -PassThru -WindowStyle Hidden
try {
    $ready = $false
    foreach ($i in 1..30) {
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2
            $ready = $true
            break
        } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $ready) { throw "serve did not become ready" }

    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
    Write-Host "  GET /health -> $($health | ConvertTo-Json -Compress)"

    $body = '{"features":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}'
    $pred = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body
    Write-Host "  POST /predict -> $($pred | ConvertTo-Json -Compress)"

    Write-Host "==> 4/5 inject drift + run the closed loop"
    python -m src.orchestrate --inject-drift

    Write-Host "==> 5/5 confirm serve now uses the promoted model"
    $pred2 = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body
    Write-Host "  POST /predict -> $($pred2 | ConvertTo-Json -Compress)"
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}

Write-Host ""
Write-Host "Done. See runs/ for the audit trail and models/ for the versioned registry."
