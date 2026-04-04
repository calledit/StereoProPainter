Get-ChildItem "datasets\youtube-vos\test_masks" -Directory | ForEach-Object {
    $name = $_.Name
    python inference_propainter.py `
        --save_fps 5 `
        --width 432 `
        --height 240 `
        --video "datasets\youtube-vos\JPEGImages\$name" `
        --mask "datasets\youtube-vos\test_masks\$name"
}