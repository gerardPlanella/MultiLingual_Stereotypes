$folderPath = "social_groups"  # Replace with the actual folder path
$excludeFiles = @("catalan_data.json", "croatian_data.json", "greek_data.json", "serbian_data.json")  # Add the file names you want to exclude

# Get all the files in the folder with the .json extension excluding the ones in $excludeFiles
$files = Get-ChildItem -Path $folderPath -File | Where-Object {
    $_.Extension -eq ".json" -and $excludeFiles -notcontains $_.Name
}

$totalFiles = $files.Count
$processedFiles = 0

foreach ($file in $files) {
    $processedFiles++

    $languagePath = Join-Path -Path $folderPath -ChildPath $file.Name
    $pythonScript = "python create_priors.py --language_path '$languagePath'"
    Write-Host "Processing file $processedFiles of $totalFiles: $languagePath"
    Start-Process -FilePath "python" -ArgumentList $pythonScript -Wait
}

Write-Host "Processing completed."
