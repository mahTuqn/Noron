[CmdletBinding()]
param(
    [string]$RunId = "helmet_20261103_155304_yolov8m",
    [string]$CheckpointPath = "",
    [int]$TrainingWaitTimeoutHours = 36,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DataConfig = "configs\data_retrain_all_stills.yaml"
$RunDirectory = Join-Path $ProjectRoot ("runs\detect\{0}" -f $RunId)
if ([string]::IsNullOrWhiteSpace($CheckpointPath)) {
    $CheckpointPath = Join-Path $RunDirectory "weights\best.pt"
} elseif (-not [System.IO.Path]::IsPathRooted($CheckpointPath)) {
    $CheckpointPath = Join-Path $ProjectRoot $CheckpointPath
}
$CheckpointPath = [System.IO.Path]::GetFullPath($CheckpointPath)

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$PostEvalRoot = Join-Path $ProjectRoot "results\retraining_post_eval"
$OutputDirectory = Join-Path $PostEvalRoot ("{0}_{1}" -f $RunId, $Timestamp)
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$SummaryPath = Join-Path $OutputDirectory "run_summary.json"
$MainLogPath = Join-Path $OutputDirectory "orchestration.log"

$script:State = [ordered]@{
    status = "training_wait"
    training_run_id = $RunId
    checkpoint_path = $CheckpointPath
    checkpoint_sha256 = $null
    dataset_config = $DataConfig
    output_directory = $OutputDirectory
    started_at_utc = [DateTime]::UtcNow.ToString("o")
    finished_at_utc = $null
    training_process_wait = [ordered]@{ matching_processes_seen = @(); ended_before_post_eval = $false }
    steps = [ordered]@{}
    evaluation = $null
    hard_case_predictions = $null
    video_diagnostic = $null
    failure_step = $null
    error = $null
}
$script:CurrentStep = "preflight"

function Save-RunSummary {
    $script:State | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
}

function Write-MainLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f [DateTime]::UtcNow.ToString("o"), $Message
    Add-Content -LiteralPath $MainLogPath -Value $line -Encoding UTF8
    Write-Host $line
}

function Get-ActiveRetrainingProcesses {
    $matches = @(
        Get-CimInstance -ClassName Win32_Process -Filter "Name='python.exe'" |
            Where-Object {
                $_.CommandLine -and
                $_.CommandLine -match '(?i)scripts[\\/]train\.py' -and
                $_.CommandLine -match '(?i)data_retrain_all_stills\.yaml' -and
                $_.CommandLine -match '(?i)--models\s+m(?:\s|$)'
            } |
            Select-Object ProcessId, ParentProcessId, CommandLine
    )
    return $matches
}

function Wait-ForRetrainingToEnd {
    $deadline = (Get-Date).AddHours($TrainingWaitTimeoutHours)
    $seen = @{}
    while ($true) {
        $active = @(Get-ActiveRetrainingProcesses)
        if ($active.Count -eq 0) {
            $script:State.training_process_wait.ended_before_post_eval = $true
            break
        }
        foreach ($process in $active) {
            $seen[[string]$process.ProcessId] = [ordered]@{
                process_id = $process.ProcessId
                parent_process_id = $process.ParentProcessId
                command_line = $process.CommandLine
            }
        }
        $script:State.training_process_wait.matching_processes_seen = @($seen.Values)
        Save-RunSummary
        if ((Get-Date) -ge $deadline) {
            throw "Timed out after $TrainingWaitTimeoutHours hours waiting for the matching train.py process to end."
        }
        $ids = ($active | ForEach-Object { $_.ProcessId }) -join ", "
        Write-MainLog "Training is still active (PID $ids); checking again in $PollSeconds seconds."
        Start-Sleep -Seconds $PollSeconds
    }
    $script:State.training_process_wait.matching_processes_seen = @($seen.Values)
    Save-RunSummary
}

function Assert-NoActiveRetraining {
    $active = @(Get-ActiveRetrainingProcesses)
    if ($active.Count -gt 0) {
        $ids = ($active | ForEach-Object { $_.ProcessId }) -join ", "
        throw "A matching training process started again (PID $ids); refusing to overlap GPU evaluation."
    }
}

function Invoke-LoggedPythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$StepName,
        [Parameter(Mandatory = $true)][string]$LogName,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Assert-NoActiveRetraining
    $script:CurrentStep = $StepName
    $logPath = Join-Path $OutputDirectory $LogName
    $commandText = '"{0}" {1}' -f $PythonExe, ($Arguments -join " ")
    Set-Content -LiteralPath $logPath -Value ("Command: " + $commandText) -Encoding UTF8
    $script:State.steps[$StepName] = [ordered]@{
        status = "in_progress"
        command = $commandText
        log_path = $logPath
        started_at_utc = [DateTime]::UtcNow.ToString("o")
        finished_at_utc = $null
        exit_code = $null
    }
    Save-RunSummary
    Write-MainLog "Starting $StepName. Log: $logPath"
    & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append | Out-Host
    $exitCode = $LASTEXITCODE
    $script:State.steps[$StepName].finished_at_utc = [DateTime]::UtcNow.ToString("o")
    $script:State.steps[$StepName].exit_code = $exitCode
    if ($exitCode -ne 0) {
        $script:State.steps[$StepName].status = "failed"
        Save-RunSummary
        throw "$StepName failed with process exit code $exitCode. See $logPath"
    }
    $script:State.steps[$StepName].status = "complete"
    Save-RunSummary
    Write-MainLog "Completed $StepName."
}

function Get-NewResultDirectory {
    param([string]$RootPath, [string[]]$NamesBefore, [string]$NamePattern)
    $candidate = Get-ChildItem -LiteralPath $RootPath -Directory -ErrorAction Stop |
        Where-Object { $_.Name -like $NamePattern -and $_.Name -notin $NamesBefore } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if (-not $candidate) {
        throw "Could not find a new result directory matching '$NamePattern' under $RootPath."
    }
    return $candidate.FullName
}

try {
    Save-RunSummary
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw "Project Python environment not found: $PythonExe"
    }
    if (-not (Test-Path -LiteralPath $DataConfig -PathType Leaf)) {
        throw "Dataset config not found: $DataConfig"
    }
    if (-not (Test-Path -LiteralPath "data\retraining_all_stills\images\test" -PathType Container)) {
        throw "Locked retraining test split is missing: data\retraining_all_stills\images\test"
    }
    if (-not (Test-Path -LiteralPath "results\external_data_analysis\video_inventory.csv" -PathType Leaf)) {
        throw "Video inventory is missing: results\external_data_analysis\video_inventory.csv"
    }

    Write-MainLog "Waiting for active training run $RunId to end before any GPU inference."
    Wait-ForRetrainingToEnd
    $script:CurrentStep = "checkpoint_check"
    if (-not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) {
        throw "Training ended, but best.pt is missing: $CheckpointPath"
    }
    if ((Get-Item -LiteralPath $CheckpointPath).Length -le 0) {
        throw "Checkpoint exists but is empty: $CheckpointPath"
    }
    $script:State.checkpoint_sha256 = (Get-FileHash -LiteralPath $CheckpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $script:State.status = "post_eval_in_progress"
    Save-RunSummary

    # Step 1: evaluate.py defaults to split=test and writes metrics_summary.json plus plots.
    $evalRoot = Join-Path $ProjectRoot "results"
    $evalPattern = ("{0}_test_*" -f $RunId)
    $evalNamesBefore = @(Get-ChildItem -LiteralPath $evalRoot -Directory | Where-Object { $_.Name -like $evalPattern } | ForEach-Object { $_.Name })
    Invoke-LoggedPythonStep -StepName "test_evaluation" -LogName "test_evaluation.log" -Arguments @(
        "scripts/evaluate.py", "--weights", $CheckpointPath, "--data", $DataConfig,
        "--device", "0", "--imgsz", "512", "--batch", "1", "--workers", "0"
    )
    $evalDirectory = Get-NewResultDirectory -RootPath $evalRoot -NamesBefore $evalNamesBefore -NamePattern $evalPattern
    $evalSummaryPath = Join-Path $evalDirectory "metrics_summary.json"
    $perClassPath = Join-Path $evalDirectory "per_class_metrics.csv"
    if (-not (Test-Path -LiteralPath $evalSummaryPath -PathType Leaf) -or -not (Test-Path -LiteralPath $perClassPath -PathType Leaf)) {
        throw "Evaluation exited successfully but expected metrics files are missing in $evalDirectory"
    }
    $evalSummary = Get-Content -LiteralPath $evalSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $script:State.evaluation = [ordered]@{
        result_directory = $evalDirectory
        metrics_summary = $evalSummaryPath
        per_class_metrics = $perClassPath
        split = $evalSummary.split
        precision = $evalSummary.precision
        recall = $evalSummary.recall
        mAP50 = $evalSummary.mAP50
        mAP50_95 = $evalSummary.mAP50_95
    }
    Save-RunSummary

    # Step 2: qualitative predictions on the two mp4-266 negatives and PPE 00751 review case.
    $hardRoot = Join-Path $ProjectRoot "results\retraining_hard_cases"
    New-Item -ItemType Directory -Path $hardRoot -Force | Out-Null
    $hardNamesBefore = @(Get-ChildItem -LiteralPath $hardRoot -Directory | ForEach-Object { $_.Name })
    Invoke-LoggedPythonStep -StepName "hard_case_predictions" -LogName "hard_case_predictions.log" -Arguments @(
        "scripts/analyze_retraining_hard_cases.py", "--weights", $CheckpointPath,
        "--output-root", "results/retraining_hard_cases"
    )
    $hardDirectory = Get-NewResultDirectory -RootPath $hardRoot -NamesBefore $hardNamesBefore -NamePattern "*"
    $hardSummaryPath = Join-Path $hardDirectory "summary.json"
    if (-not (Test-Path -LiteralPath $hardSummaryPath -PathType Leaf)) {
        throw "Hard-case prediction summary is missing: $hardSummaryPath"
    }
    $hardSummary = Get-Content -LiteralPath $hardSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($hardSummary.cases.Count -ne 3) {
        throw "Expected 3 user-reviewed hard cases, found $($hardSummary.cases.Count)."
    }
    $script:State.hard_case_predictions = [ordered]@{
        output_directory = $hardDirectory
        summary = $hardSummaryPath
        case_count = $hardSummary.cases.Count
        predictions_are_ground_truth = $false
        scoring_performed = $false
    }
    Save-RunSummary

    # Step 3: frame-by-frame inference for all 32 clips; no annotated MP4s are encoded.
    $videoRoot = Join-Path $ProjectRoot "results\motorcycle_video_diagnostic"
    New-Item -ItemType Directory -Path $videoRoot -Force | Out-Null
    $videoNamesBefore = @(Get-ChildItem -LiteralPath $videoRoot -Directory | ForEach-Object { $_.Name })
    Invoke-LoggedPythonStep -StepName "video_diagnostic" -LogName "video_diagnostic.log" -Arguments @(
        "scripts/infer_motorcycle_video_dataset.py", "--weights", $CheckpointPath,
        "--output-root", "results/motorcycle_video_diagnostic",
        "--inventory", "results/external_data_analysis/video_inventory.csv",
        "--video-root", $ProjectRoot, "--conf", "0.25", "--imgsz", "512",
        "--device", "0", "--expected-videos", "32"
    )
    $videoDirectory = Get-NewResultDirectory -RootPath $videoRoot -NamesBefore $videoNamesBefore -NamePattern "*"
    $videoSummaryPath = Join-Path $videoDirectory "summary.json"
    $detectionsPath = Join-Path $videoDirectory "frame_detections.jsonl"
    if (-not (Test-Path -LiteralPath $videoSummaryPath -PathType Leaf) -or -not (Test-Path -LiteralPath $detectionsPath -PathType Leaf)) {
        throw "Video diagnostic outputs are incomplete in $videoDirectory"
    }
    $videoSummary = Get-Content -LiteralPath $videoSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($videoSummary.overall.videos -ne 32 -or
        $videoSummary.overall.frames_expected_from_inventory -ne 114640 -or
        $videoSummary.overall.frames_decoded -ne 114640 -or
        $videoSummary.frame_count_mismatches.Count -ne 0) {
        throw "Video diagnostic did not complete all 32 videos / 114,640 frames; inspect $videoSummaryPath"
    }
    $script:State.video_diagnostic = [ordered]@{
        output_directory = $videoDirectory
        summary = $videoSummaryPath
        frame_detections = $detectionsPath
        videos = $videoSummary.overall.videos
        frames_decoded = $videoSummary.overall.frames_decoded
        frames_with_detections = $videoSummary.overall.frames_with_detections
        detection_count = $videoSummary.overall.detection_count
        frame_count_mismatches = $videoSummary.frame_count_mismatches.Count
        supervised_metrics_available = $false
    }
    $script:State.status = "post_eval_complete"
    $script:State.finished_at_utc = [DateTime]::UtcNow.ToString("o")
    Save-RunSummary
    Write-MainLog "Post-training evaluation completed. Summary: $SummaryPath"
    Write-Host "Run summary: $SummaryPath"
}
catch {
    $script:State.status = "failed"
    $script:State.failure_step = $script:CurrentStep
    $script:State.error = $_.Exception.Message
    $script:State.finished_at_utc = [DateTime]::UtcNow.ToString("o")
    Save-RunSummary
    Write-MainLog "FAILED at $($script:CurrentStep): $($script:State.error)"
    throw
}
