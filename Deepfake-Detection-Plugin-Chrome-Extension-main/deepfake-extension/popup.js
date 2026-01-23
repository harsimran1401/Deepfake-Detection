document.getElementById('scanBtn').addEventListener('click', async () => {
    const resultDiv = document.getElementById('result');
    
    // UI Feedback
    resultDiv.innerHTML = "Scanning... ⏳";
    resultDiv.className = "loading";

    // 1. Get the current active tab
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // 2. Ask content.js to grab the frame
    chrome.tabs.sendMessage(tab.id, { action: "capture_frame" }, async (response) => {
        
        // Handle connection errors
        if (chrome.runtime.lastError) {
            resultDiv.innerHTML = "Please refresh the page and try again.";
            resultDiv.className = "error";
            return;
        }

        if (!response || response.error) {
            resultDiv.innerHTML = response.error || "Error capturing video.";
            resultDiv.className = "error";
            return;
        }

        if (response.imageData) {
            // 3. Send the image to your Flask Backend
            try {
                console.log("Sending image to server...");
                
                const apiResponse = await fetch('http://127.0.0.1:5000/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: response.imageData })
                });

                const data = await apiResponse.json();

                // Handle "No Face" case
                if (data.verdict === "No Face Detected") {
                    resultDiv.innerHTML = `🤷 No Face Found<br><small>Try pausing on a clear face.</small>`;
                    resultDiv.className = "loading"; 
                } 
                // Handle Real
                else if (data.verdict === "real" || data.verdict === "REAL") {
                    resultDiv.innerHTML = `✅ Looks Authentic<br><small>Confidence: ${data.confidence}%</small>`;
                    resultDiv.className = "safe";
                } 
                // Handle Fake
                else {
                    resultDiv.innerHTML = `⚠️ Potential Deepfake<br><small>Confidence: ${data.confidence}%</small>`;
                    resultDiv.className = "danger";
                }

            } catch (error) {
                // This catches server errors (like if app.py is off)
                resultDiv.innerHTML = "❌ Error connecting to server.<br><small>Is app.py running?</small>";
                resultDiv.className = "error";
                console.error(error);
            }
        } // End of if(response.imageData)

    }); // End of sendMessage
}); // End of addEventListener

