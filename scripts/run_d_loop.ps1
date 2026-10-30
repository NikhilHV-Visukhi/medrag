# D (generation eval): keep making resume passes until every answer is judged.
# Gemini's free-tier judge quota comes back in bursts, so wait between passes when answers are still unjudged.
# -Only cloud | local runs one half, so the local (GPU) models can run alongside the cloud ones.
param([string]$Only = 'both')
Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONIOENCODING = 'utf-8'
$env:JUDGE_MODEL = 'gemini-3.6-flash'
$py = '.venv\Scripts\python.exe'
$common = '--skip-retrieval', '--per-type', '20', '--max-answers', '54', '--judge'
$cloud = 'gemini:gemini-3.5-flash,gemini:gemini-3.5-flash-lite,gemini:gemini-3.1-flash-lite,gemini:gemini-3-flash-preview'
$local = 'ollama:qwen3:1.7b,ollama:qwen2.5:3b'

function Judged($file) {
    if (-not (Test-Path $file)) { return 0 }
    $ok = @{}
    foreach ($line in Get-Content $file -Encoding utf8) {
        $r = $line | ConvertFrom-Json
        if ($null -ne $r.judge_correctness -and $r.judge -eq "gemini:$($env:JUDGE_MODEL)") { $ok["$($r.llm)|$($r.id)"] = 1 }
    }
    return $ok.Count
}

$doCloud = $Only -ne 'local'; $doLocal = $Only -ne 'cloud'
function Left { ($doCloud -and (Judged 'reports\gen_large.jsonl') -lt 216) -or ($doLocal -and (Judged 'reports\gen_large_local.jsonl') -lt 108) }

for ($pass = 1; $pass -le 60; $pass++) {
    "$(Get-Date -Format 'HH:mm') [$Only] pass $pass  judged: cloud $(Judged 'reports\gen_large.jsonl')/216, local $(Judged 'reports\gen_large_local.jsonl')/108"
    if (-not (Left)) { 'all judged'; break }
    if ($doCloud -and (Judged 'reports\gen_large.jsonl') -lt 216) { & $py -m medrag eval @common --llms $cloud --resume reports/gen_large.jsonl *> reports\eval_gen_cloud_gpu.log }
    if ($doLocal -and (Judged 'reports\gen_large_local.jsonl') -lt 108) { & $py -m medrag eval @common --llms $local --resume reports/gen_large_local.jsonl *> reports\eval_gen_local_gpu.log }
    if (Left) { Start-Sleep -Seconds 300 }
}
