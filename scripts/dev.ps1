param(
    [ValidateSet("start", "stop", "restart", "status")]
    [string]$Action = "restart"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RuntimeDir = Join-Path $ProjectRoot ".runtime"
$PidFile = Join-Path $RuntimeDir "dev-processes.json"
$LogDir = Join-Path $ProjectRoot "logs"
$ApiPort = 8000
$UiPort = 8501
$ApiUrl = "http://127.0.0.1:$ApiPort"

function Get-ManagedProcessIds {
    $processes = @(Get-CimInstance Win32_Process)
    $rootPattern = [regex]::Escape((Join-Path $ProjectRoot ".venv\Scripts"))
    $managedCommand = 'uvicorn\s+api\.server:app|streamlit(?:\.exe)?"?\s+(?:run|-m\s+streamlit\s+run)|-m\s+streamlit\s+run'
    $ids = [System.Collections.Generic.HashSet[int]]::new()

    foreach ($process in $processes) {
        if ($process.CommandLine -match $rootPattern -and $process.CommandLine -match $managedCommand) {
            [void]$ids.Add([int]$process.ProcessId)
        }
    }

    if (Test-Path $PidFile) {
        try {
            $saved = Get-Content $PidFile -Raw | ConvertFrom-Json
            foreach ($id in @($saved.api_pid, $saved.ui_pid)) {
                if ($id) { [void]$ids.Add([int]$id) }
            }
        } catch {
            Write-Warning "Ignoring invalid process file: $PidFile"
        }
    }

    do {
        $added = $false
        foreach ($process in $processes) {
            if ($ids.Contains([int]$process.ParentProcessId) -and $ids.Add([int]$process.ProcessId)) {
                $added = $true
            }
        }
    } while ($added)

    return @($ids)
}

function Stop-LocalServices {
    $ids = @(Get-ManagedProcessIds)
    if ($ids.Count -gt 0) {
        Write-Host "Stopping project services: $($ids -join ', ')"
        Stop-Process -Id $ids -Force -ErrorAction SilentlyContinue
        $deadline = (Get-Date).AddSeconds(10)
        while ((Get-Date) -lt $deadline -and (Get-Process -Id $ids -ErrorAction SilentlyContinue)) {
            Start-Sleep -Milliseconds 200
        }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

function Assert-PortAvailable([int]$Port) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if ($listener) {
        throw "Port $Port is occupied by PID $($listener[0].OwningProcess). Stop that process before starting this project."
    }
}

function Wait-HttpReady([string]$Url, [string]$Name) {
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Write-Host "$Name ready: $Url"
                return
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "$Name did not become ready within 60 seconds. Check $LogDir."
}

function Start-LocalServices {
    if (-not (Test-Path $Python)) {
        throw "Virtual environment not found: $Python"
    }
    Assert-PortAvailable $ApiPort
    Assert-PortAvailable $UiPort
    New-Item -ItemType Directory -Force -Path $RuntimeDir, $LogDir | Out-Null

    $api = Start-Process -FilePath $Python -ArgumentList @(
        "-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", "$ApiPort"
    ) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogDir "api.out.log") -RedirectStandardError (Join-Path $LogDir "api.err.log") -PassThru

    try {
        Wait-HttpReady "$ApiUrl/health" "FastAPI"
        $env:AGENT_API_BASE_URL = $ApiUrl
        $ui = Start-Process -FilePath $Python -ArgumentList @(
            "-m", "streamlit", "run", "app.py",
            "--server.address", "127.0.0.1", "--server.port", "$UiPort", "--server.headless", "true"
        ) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogDir "ui.out.log") -RedirectStandardError (Join-Path $LogDir "ui.err.log") -PassThru
        Wait-HttpReady "http://127.0.0.1:$UiPort/" "Streamlit"
    } catch {
        Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
        if ($ui) { Stop-Process -Id $ui.Id -Force -ErrorAction SilentlyContinue }
        throw
    }

    @{
        api_pid = $api.Id
        ui_pid = $ui.Id
        api_url = $ApiUrl
        ui_url = "http://127.0.0.1:$UiPort"
        started_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8
    Write-Host "Local services started. UI: http://127.0.0.1:$UiPort"
}

function Show-LocalStatus {
    foreach ($port in @($ApiPort, $UiPort)) {
        $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
        if ($listener) {
            Write-Host "Port $port listening (PID $($listener[0].OwningProcess))"
        } else {
            Write-Host "Port $port stopped"
        }
    }
}

Set-Location $ProjectRoot
switch ($Action) {
    "start" { Start-LocalServices }
    "stop" { Stop-LocalServices; Show-LocalStatus }
    "restart" { Stop-LocalServices; Start-LocalServices }
    "status" { Show-LocalStatus }
}
