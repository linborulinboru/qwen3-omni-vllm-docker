# transcribe.ps1 - Batch transcribe all media files in app\inputs
# Uses docker exec to run qwen3_omni.py inside the container

$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$inputsDir  = Join-Path $scriptDir "app\inputs"
$outputsDir = Join-Path $scriptDir "app\outputs"
$container  = "qwen3-omni-serve"
$extensions = @("mp4","mp3","wav","m4a","mkv","avi","mov","flac","ogg","webm")

if (-not (Test-Path $outputsDir)) {
    New-Item -ItemType Directory -Path $outputsDir | Out-Null
}

Write-Host "============================================================"
Write-Host " Qwen3-Omni Batch Transcription (docker exec)"
Write-Host "============================================================"
Write-Host " Inputs:    $inputsDir"
Write-Host " Outputs:   $outputsDir"
Write-Host " Container: $container"
Write-Host "============================================================"
Write-Host ""

# Check container is running
$state = docker inspect --format "{{.State.Running}}" $container 2>$null
if ($state -ne "true") {
    Write-Host "[ERROR] Container '$container' is not running."
    Write-Host "        Start it with:  docker compose up -d"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK]  Container is running."
Write-Host ""

# Wait until API is healthy (serve layer + vLLM backend)
Write-Host "[INFO] Waiting for API to become ready..."
while ($true) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:5000/health" -Method Get -TimeoutSec 10 -ErrorAction Stop
        if ($health.status -eq "ok" -and $health.vllm_connected -eq $true) {
            Write-Host "[OK]  API healthy. vLLM model: $($health.vllm_model)"
            break
        } else {
            Write-Host "[WAIT] API degraded (vllm_connected=$($health.vllm_connected)). Retrying in 10s..."
        }
    } catch {
        Write-Host "[WAIT] API not reachable: $_. Retrying in 10s..."
    }
    Start-Sleep -Seconds 10
}
Write-Host ""

$found   = 0
$success = 0
$failed  = 0

foreach ($ext in $extensions) {
    $files = Get-ChildItem -Path $inputsDir -Filter "*.$ext" -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        $found++
        $containerPath = "/app/inputs/$($file.Name)"

        Write-Host "[INFO] Processing: $($file.Name)"
        Write-Host "[INFO] Output:     $($file.BaseName).txt"

        docker exec $container python -u /app/serve/qwen3_omni.py --file $containerPath

        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK]  Saved: app\outputs\$($file.BaseName).txt"
            $success++
        } else {
            Write-Host "[ERR] Failed: $($file.Name)"
            $failed++
        }
        Write-Host ""
    }
}

if ($found -eq 0) {
    Write-Host "[WARN] No media files found in: $inputsDir"
    Write-Host "       Supported formats: $($extensions -join ', ')"
    Write-Host ""
}

Write-Host "============================================================"
Write-Host " Done.  Success: $success   Failed: $failed"
Write-Host "============================================================"
Read-Host "Press Enter to exit"
