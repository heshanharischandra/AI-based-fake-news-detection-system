$pythonPath = Join-Path $PSScriptRoot ".venv311/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Error "Python environment not found at $pythonPath"
    exit 1
}

Write-Host "Add verified evidence for image context scoring" -ForegroundColor Cyan
Write-Host "Use a real publisher or fact-checker URL. Do not enter guessed facts." -ForegroundColor Yellow

$imagePath = Read-Host "Full path of the reference image"
$sourceName = Read-Host "Publisher or fact-checker name"
$sourceUrl = Read-Host "Canonical source URL"
$publicationDate = Read-Host "Publication date (YYYY-MM-DD)"
$eventDate = Read-Host "Original event date (YYYY-MM-DD)"
$eventName = Read-Host "Original event name"
$location = Read-Host "Original event location"
$originalClaim = Read-Host "Original documented claim/headline"
$filename = Read-Host "Optional saved image name (press Enter to keep the original name)"

$arguments = @(
    "scripts/register_verified_reference.py",
    "--image", $imagePath,
    "--source-name", $sourceName,
    "--source-url", $sourceUrl,
    "--publication-date", $publicationDate,
    "--event-date", $eventDate,
    "--event-name", $eventName,
    "--location", $location,
    "--original-claim", $originalClaim
)
if ($filename) {
    $arguments += @("--filename", $filename)
}

& $pythonPath @arguments
