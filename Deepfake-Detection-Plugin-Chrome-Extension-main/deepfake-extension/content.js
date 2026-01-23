// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    
    if (request.action === "capture_frame") {
        console.log("Deepfake Detector: Attempting to capture frame...");
        
        // Find the video element on the page
        const video = document.querySelector('video');

        if (video) {
            // Check if video is actually playing or has data
            if (video.readyState < 2) {
                sendResponse({ error: "Video not ready or not loaded." });
                return true;
            }

            try {
                // 1. Create a canvas to draw the frame
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');

                // 2. Draw the current video frame onto the canvas
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // 3. Convert to Base64 image string (this is what we send to Python)
                const dataUrl = canvas.toDataURL('image/jpeg');
                
                console.log("Deepfake Detector: Frame captured successfully!");
                sendResponse({ imageData: dataUrl });

            } catch (e) {
                // Security error can happen on Netflix/Disney+ due to DRM
                console.error("Deepfake Detector Error:", e);
                sendResponse({ error: "Cannot capture this video (DRM Protected?)" });
            }
        } else {
            console.log("Deepfake Detector: No video element found.");
            sendResponse({ error: "No video found on this page." });
        }
    }
    // Return true to indicate we will send a response asynchronously
    return true; 
});