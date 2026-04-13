param(
    [string]$SshHost = "root@px-cloud2.matpool.com",
    [int]$Port = 29196,
    [string]$RemoteResultsRoot = "/mnt/results/ablation",
    [string]$LocalArchiveRoot = "C:\Document\graduationProject\openpangu-embedded-7b-model\experiments\results\archives",
    [int]$PollSeconds = 300,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param(
        [string]$Message
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Add-Content -Path $script:LogPath -Value $line -Encoding UTF8
    Write-Output $line
}

function Write-Status {
    param(
        [hashtable]$Payload
    )

    $Payload["updated_at"] = (Get-Date).ToString("o")
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -Path $script:StatusPath -Encoding UTF8
}

function Convert-ToScpPath {
    param(
        [string]$PathValue
    )

    return $PathValue -replace "\\", "/"
}

function Invoke-RemoteScript {
    param(
        [string]$ScriptText
    )

    $normalizedScript = $ScriptText -replace "`r`n", "`n" -replace "`r", "`n"
    $output = $normalizedScript | & ssh -T -p $Port $SshHost "bash -s" 2>&1
    $exitCode = $LASTEXITCODE
    $text = ($output | Out-String).Trim()
    if ($exitCode -ne 0) {
        throw "Remote command failed with exit code ${exitCode}: $text"
    }
    return $text
}

function Parse-KeyValueText {
    param(
        [string]$Text
    )

    $map = @{}
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -match "^(?<key>[A-Z0-9_]+)=(?<value>.*)$") {
            $map[$matches["key"]] = $matches["value"]
        }
    }
    return $map
}

function Get-RemoteQueueState {
    $remoteScript = @'
set -euo pipefail
results_root='__REMOTE_RESULTS_ROOT__'
latest_log=$(ls -1t "$results_root"/ablation_master_queue_*.log 2>/dev/null | head -n 1 || true)
queue_present=0
if pgrep -af "bash $results_root/run_all_ablation_studies.sh" >/dev/null 2>&1; then
  queue_present=1
fi
runner_count=$( (pgrep -af "run_sparsegpt_scaffold.py|run_admm_scaffold.py|run_gptq_scaffold.py|run_awq_scaffold.py|run_smoothquant_scaffold.py|run_ablation_manifest.py|run_ablation_variant_pipeline.py|run_final_artifact_model_benchmark.py|run_reload_verification.py|run_compressed_artifact_verification.py" 2>/dev/null || true) | wc -l | tr -d ' ' )
done_flag=0
if [ -n "$latest_log" ] && grep -q "\[DONE\]" "$latest_log"; then
  done_flag=1
fi
printf 'LATEST_LOG=%s\nQUEUE_PRESENT=%s\nRUNNER_COUNT=%s\nDONE_FLAG=%s\n' "$latest_log" "$queue_present" "$runner_count" "$done_flag"
'@
    $remoteScript = $remoteScript.Replace("__REMOTE_RESULTS_ROOT__", $RemoteResultsRoot)
    return (Parse-KeyValueText -Text (Invoke-RemoteScript -ScriptText $remoteScript))
}

function New-RemoteBundle {
    $remoteScript = @'
set -euo pipefail
results_root='__REMOTE_RESULTS_ROOT__'
bundle_name="ablation_sync_bundle_$(date +%Y%m%d-%H%M%S).tar.gz"
file_list="/tmp/ablation-sync-files-$$.txt"
cleanup() {
  rm -f "$file_list"
}
trap cleanup EXIT
cd "$results_root"
find . -type f \( -name '*.json' -o -name '*.md' -o -name '*.log' -o -name '*.sh' -o -name '*.pid' -o -name '*.txt' \) \
  ! -path '*/exported_model/*' \
  ! -path '*/compressed_artifact/*' | sort > "$file_list"
tar -czf "$results_root/$bundle_name" -T "$file_list"
file_count=$(wc -l < "$file_list" | tr -d ' ')
sha256=$(sha256sum "$results_root/$bundle_name" | awk '{print $1}')
printf 'BUNDLE_PATH=%s\nFILE_COUNT=%s\nSHA256=%s\n' "$results_root/$bundle_name" "$file_count" "$sha256"
'@
    $remoteScript = $remoteScript.Replace("__REMOTE_RESULTS_ROOT__", $RemoteResultsRoot)
    return (Parse-KeyValueText -Text (Invoke-RemoteScript -ScriptText $remoteScript))
}

function Remove-RemoteFile {
    param(
        [string]$RemotePath
    )

    $remoteScript = @'
set -euo pipefail
rm -f '__REMOTE_PATH__'
'@
    $remoteScript = $remoteScript.Replace("__REMOTE_PATH__", $RemotePath)
    [void](Invoke-RemoteScript -ScriptText $remoteScript)
}

New-Item -ItemType Directory -Force -Path $LocalArchiveRoot | Out-Null

$sessionStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$script:SessionRoot = Join-Path $LocalArchiveRoot "ablation_sync_$sessionStamp"
$script:LogPath = Join-Path $LocalArchiveRoot "ablation_monitor_$sessionStamp.log"
$script:StatusPath = Join-Path $LocalArchiveRoot "ablation_monitor_status.json"
$lockPath = Join-Path $LocalArchiveRoot "ablation_monitor.lock"
$startedAt = (Get-Date).ToString("o")

if (-not $Once) {
    if (Test-Path -LiteralPath $lockPath) {
        $existingPidText = (Get-Content -Path $lockPath -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($existingPidText -match "^\d+$") {
            $existingProcess = Get-Process -Id ([int]$existingPidText) -ErrorAction SilentlyContinue
            if ($null -ne $existingProcess) {
                throw "A monitor is already running with PID=$existingPidText"
            }
        }
    }
    Set-Content -Path $lockPath -Value $PID -Encoding ASCII
}

try {
    Write-Log "monitor_started pid=$PID once=$($Once.IsPresent) remote_root=$RemoteResultsRoot"
    Write-Status @{
        state = "monitoring"
        started_at = $startedAt
        pid = $PID
        once = $Once.IsPresent
        remote_results_root = $RemoteResultsRoot
        local_archive_root = $LocalArchiveRoot
        log_path = $script:LogPath
        session_root = $script:SessionRoot
        sync_complete = $false
    }

    $finalState = $null
    $remoteState = $null
    while ($true) {
        $remoteState = Get-RemoteQueueState
        $queuePresent = $remoteState["QUEUE_PRESENT"] -eq "1"
        $runnerCountText = $remoteState["RUNNER_COUNT"]
        if ([string]::IsNullOrWhiteSpace($runnerCountText)) {
            $runnerCountText = "0"
        }
        $runnerCount = [int]$runnerCountText
        $doneFlag = $remoteState["DONE_FLAG"] -eq "1"
        $latestLog = $remoteState["LATEST_LOG"]

        Write-Log "remote_state queue_present=$queuePresent runner_count=$runnerCount done_flag=$doneFlag latest_log=$latestLog"
        Write-Status @{
            state = "monitoring"
            started_at = $startedAt
            pid = $PID
            once = $Once.IsPresent
            remote_results_root = $RemoteResultsRoot
            local_archive_root = $LocalArchiveRoot
            log_path = $script:LogPath
            session_root = $script:SessionRoot
            remote_state = @{
                latest_log = $latestLog
                queue_present = $queuePresent
                runner_count = $runnerCount
                done_flag = $doneFlag
            }
            sync_complete = $false
        }

        $queueIdle = (-not $queuePresent) -and ($runnerCount -eq 0)
        if ($queueIdle -and $doneFlag) {
            $finalState = "completed"
            break
        }
        if ($queueIdle -and -not [string]::IsNullOrWhiteSpace($latestLog) -and -not $doneFlag) {
            $finalState = "stopped_without_done"
            break
        }
        if ($Once) {
            return
        }
        Start-Sleep -Seconds $PollSeconds
    }

    Write-Log "queue_terminal_state state=$finalState"
    New-Item -ItemType Directory -Force -Path $script:SessionRoot | Out-Null

    $bundleInfo = New-RemoteBundle
    $remoteBundlePath = $bundleInfo["BUNDLE_PATH"]
    $remoteSha256 = $bundleInfo["SHA256"]
    $fileCountText = $bundleInfo["FILE_COUNT"]
    if ([string]::IsNullOrWhiteSpace($fileCountText)) {
        $fileCountText = "0"
    }
    $fileCount = [int]$fileCountText
    if ([string]::IsNullOrWhiteSpace($remoteBundlePath)) {
        throw "Remote bundle path is empty."
    }

    $localBundleName = Split-Path -Path $remoteBundlePath -Leaf
    $localBundlePath = Join-Path $script:SessionRoot $localBundleName
    $localBundleScpPath = Convert-ToScpPath -PathValue $localBundlePath
    & scp -P $Port "${SshHost}:$remoteBundlePath" $localBundleScpPath
    if ($LASTEXITCODE -ne 0) {
        throw "scp download failed with exit code $LASTEXITCODE"
    }

    $localHash = (Get-FileHash -Algorithm SHA256 -Path $localBundlePath).Hash.ToLowerInvariant()
    if ($localHash -ne $remoteSha256.ToLowerInvariant()) {
        throw "SHA256 mismatch. local=$localHash remote=$remoteSha256"
    }

    $extractRoot = Join-Path $script:SessionRoot "bundle"
    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
    & tar -xzf $localBundlePath -C $extractRoot
    if ($LASTEXITCODE -ne 0) {
        throw "tar extract failed with exit code $LASTEXITCODE"
    }

    Remove-RemoteFile -RemotePath $remoteBundlePath

    Write-Status @{
        state = $finalState
        started_at = $startedAt
        pid = $PID
        once = $Once.IsPresent
        remote_results_root = $RemoteResultsRoot
        local_archive_root = $LocalArchiveRoot
        log_path = $script:LogPath
        session_root = $script:SessionRoot
        remote_state = @{
            latest_log = $remoteState["LATEST_LOG"]
            queue_present = $false
            runner_count = 0
            done_flag = ($finalState -eq "completed")
        }
        sync_complete = $true
        sync = @{
            remote_bundle_path = $remoteBundlePath
            remote_sha256 = $remoteSha256
            local_bundle_path = $localBundlePath
            local_extract_root = $extractRoot
            file_count = $fileCount
        }
    }

    Write-Log "sync_complete state=$finalState file_count=$fileCount local_bundle=$localBundlePath"
}
catch {
    Write-Log "monitor_error $($_.Exception.Message)"
    Write-Status @{
        state = "error"
        started_at = $startedAt
        pid = $PID
        once = $Once.IsPresent
        remote_results_root = $RemoteResultsRoot
        local_archive_root = $LocalArchiveRoot
        log_path = $script:LogPath
        session_root = $script:SessionRoot
        sync_complete = $false
        error = $_.Exception.Message
    }
    throw
}
finally {
    if ((-not $Once) -and (Test-Path -LiteralPath $lockPath)) {
        Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    }
}
