// ダウンロード速度テスト
function testDownloadSpeed(downloadUrl) {
    const startTime = Date.now();
    fetch(downloadUrl)
        .then(response => response.blob())
        .then(blob => {
            const endTime = Date.now();
            const duration = (endTime - startTime) / 1000; // 秒単位
            const fileSizeInBytes = blob.size;
            const speedInBps = (fileSizeInBytes * 8) / duration;
            const speedInMbps = speedInBps / 1024 / 1024;
            alert(`Download Speed: ${speedInMbps.toFixed(2)} Mbps`);
        })
        .catch(error => alert('Download test failed: ' + error.message));
}

// アップロード速度テスト
function testUploadSpeed(uploadUrl, dataSizeInMB = 1) {
    const data = new Blob([new Uint8Array(dataSizeInMB * 1024 * 1024)], { type: 'application/octet-stream' });
    const startTime = Date.now();
    fetch(uploadUrl, {
        method: 'POST',
        body: data,
    })
    .then(response => response.text())
    .then(result => {
        const endTime = Date.now();
        const duration = (endTime - startTime) / 1000; // 秒単位
        const speedInBps = (dataSizeInMB * 1024 * 1024 * 8) / duration;
        const speedInMbps = speedInBps / 1024 / 1024;
        alert(`Upload Speed: ${speedInMbps.toFixed(2)} Mbps`);
    })
    .catch(error => alert('Upload test failed: ' + error.message));
}

document.getElementById('startTest').addEventListener('click', function() {
    const downloadTestFileUrl = 'https://github.com/takerusQ/internet-speed-test/blob/main/dummyfile10mb_for_networktest'; // 大きなファイルのURL
    const uploadTestEndpoint = 'YOUR_HEROKU_OR_FIREBASE_ENDPOINT_HERE'; // HerokuやFirebaseで設定したアップロードエンドポイントを指定してください。
    testDownloadSpeed(downloadTestFileUrl);
    testUploadSpeed(uploadTestEndpoint);
});
