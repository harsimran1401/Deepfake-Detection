## Deepfake-Detection-Plugin-Chrome Extension
This repository houses the code for building your own deepfake detection plugin (chrome extension) MVP developed by Bashaar Naik on 9th January, 2025!

**Deepfake Guard** is a real-time browser extension designed to detect AI-generated faces in video streams (YouTube, etc.) using a multi-model ensemble approach.

## 🚀 Key Features
- **Real-Time Analysis**: Captures frames directly from the active browser tab.
- **Ensemble AI Core**: Combines 4 distinct detection algorithms (CNN, ViT, CLIP, CV).
- **Transparent Breakdown**: Displays individual confidence scores for every model in a neat table.
- **Visual Dashboard**: Color-coded "Real" vs "Fake" badges with confidence percentages.

**Demonstration**



![Picture2](https://github.com/user-attachments/assets/69e1bf88-c9c8-460e-aaa3-735ea0f05f6e)
![Picture1](https://github.com/user-attachments/assets/4d8daa2f-9cb9-4c27-b511-ca61af80fdf9)


## 📦 Installation
**1. Backend Setup (The Brain)**

   - **Clone this Repository**
   - **Go to deepfake-backend and run the server_master.py file.**


**2. Extension Installation**

  - **Open Chrome and navigate to chrome://extensions.**
  
  - **Enable "Developer mode" (toggle in the top-right corner).**
  
  - **Click the "Load unpacked" button.**
  
  - **Select the deepfake-extension folder inside the project directory.**


## 📂 Project Structure
Here is how the project files are organized:

```text
/DeepFake Detection Plugin
│
├── /deepfake-backend          # 🗄️ Backend Server and Model Hosting
│   ├── server_master.py       
│   ├── server_ensemble.py         
│   ├── server_effnet.py
│   ├── server_openaiclip.py
│   ├── server_resnet.py
│   ├── server_vit.py
│   └── server.py
│
├── requirements.txt           # 📦 Dependencies list for Python
│
├── /deepfake-extension        # 👁️ THE EYES: Chrome Extension Folder
│   ├── manifest.json          #    Config file (Permissions, Version)
│   ├── popup.html             #    The User Interface (Buttons, Table)
│   ├── popup.js               #    Logic (Talking to Python, Updating UI)
│   ├── content.js             #    Script injected into YouTube to capture video
│   └── icon.png               #    (Optional) Extension Icon
│
└── README.md                  #    This documentation file
└── Explanation.pdf           #    Explaining the overview of the project and demonstrating the prototype


