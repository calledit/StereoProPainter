

conda activate propainter

python scripts/compute_flow.py --root_path datasets\youtube-vos\JPEGImages --save_path your_flow_root --height 240 --width 432

cd datasets\youtube-vos\JPEGImages

conda activate mdvt
Get-ChildItem -Directory | ForEach-Object {
    $folder = $_.Name
    $listFile = "$folder\filelist.txt"
    $outputFile = "..\mkvs\$folder.mkv"

    # Kontrollera om filen redan finns
    if (Test-Path -Path $outputFile) {
        Write-Host "Hoppar över: $folder (mkv finns redan)" -ForegroundColor Yellow
    } else {
        Write-Host "Processar: $folder" -ForegroundColor Cyan

        # Skapa listan med filer
        Get-ChildItem -Path $folder -Filter *.jpg | Sort-Object Name | ForEach-Object {
            "file '$($_.Name)'"
        } | Out-File -FilePath $listFile -Encoding ascii

        # Kör FFmpeg
        ffmpeg -y -r 5 -f concat -safe 0 -i $listFile -c:v ffv1 "$outputFile"

        # Ta bort den temporära listan
        if (Test-Path $listFile) { Remove-Item $listFile }
    }
}


cd ..\mkvs\

# Sökvägen till ditt skript
$pythonScript = "C:\Users\calle\projects\metric_depth_video_toolbox\video_da3.py"


$listFile = Join-Path $pwd "video_list.txt"

Remove-Item $listFile

# Loopa igenom alla .mkv-filer i den nuvarande mappen
Get-ChildItem -Filter *.mkv | ForEach-Object {
    $videoFile = $_.Name
    #$baseName = $_.BaseName # Filnamnet utan .mkv
    $depthFileName = "${videoFile}_depth.mkv"
	
	$fullPath  = $_.FullName

    # 1. Kolla om filen själv slutar på _depth.mkv
    # 2. Kolla om en fil med namnet _depth.mkv redan existerar
    if ($videoFile -like "*_depth.mkv") {
        Write-Host "Skippar: $videoFile (är redan en djupfil)" -ForegroundColor Yellow
    }
    elseif (Test-Path $depthFileName) {
        Write-Host "Skippar: $videoFile (djupfil finns redan: $depthFileName)" -ForegroundColor Cyan
    }
    else {
		# Lägg till filnamnet i textfilen
        [System.IO.File]::AppendAllText($listFile, "$fullPath`n")
        Write-Host "Lagt till: $videoFile" -ForegroundColor Green
    }
}

# Om listan skapades och inte är tom, kör Python-skriptet en gång
if (Test-Path $listFile) {
    Write-Host "`nKör Python-skriptet med alla filer i $listFile..." -ForegroundColor Magenta
    python $pythonScript --color_video $listFile
	
	Remove-Item $listFile
}else {
    Write-Host "`nInga nya videofiler att bearbeta." -ForegroundColor Yellow
}


cd ..
cd ..
cd ..


python generate_traning_data_from_dataset.py