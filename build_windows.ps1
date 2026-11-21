$ErrorActionPreference = "Stop"

if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
    throw "Run this build script on Windows."
}

$ffmpeg = $env:FFMPEG_EXE
if (-not $ffmpeg) {
    $ffmpegCommand = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
    if ($ffmpegCommand) {
        $ffmpeg = $ffmpegCommand.Source
    }
}
if (-not $ffmpeg -or -not (Test-Path -LiteralPath $ffmpeg -PathType Leaf)) {
    throw "Set FFMPEG_EXE to an FFmpeg executable or add ffmpeg.exe to PATH."
}

python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --console `
    --name WhisperOverlay `
    --collect-all faster_whisper `
    --collect-all ctranslate2 `
    --collect-all av `
    --add-binary "$ffmpeg;." `
    app.py

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE."
}

Write-Host "Created dist\WhisperOverlay.exe"
