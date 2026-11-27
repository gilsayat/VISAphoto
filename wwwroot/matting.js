// Listen for the massive Base64 24MP image from C#
window.chrome.webview.addEventListener('message', async event => {
    try {
        const base64ImageIn = event.data;
        
        // Configure for maximum studio quality
        const config = {
            // FIXED: Pointing strictly to local app assets mapped by C#
            publicPath: "http://local.appassets/",
            model: "isnet_fp16", // High-precision Vision Transformer
            output: { format: "image/png", quality: 1.0 }
        };
        
        // Execute the extraction via WebGPU/WebGL
        const blob = await imglyRemoveBackground(base64ImageIn, config);
        
        // Read the flawless PNG back to Base64
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = function() {
            // Send the transparent PNG back to C#
            window.chrome.webview.postMessage(JSON.stringify({ 
                success: true, 
                data: reader.result 
            }));
        }
    } catch (e) {
        window.chrome.webview.postMessage(JSON.stringify({ 
            success: false, 
            error: e.message 
        }));
    }
});

// Signal C# that the WebView2 engine is loaded and ready
window.chrome.webview.postMessage(JSON.stringify({ success: true, status: "READY" }));
