# Installation Win64

## Setup Python
We use Python 3.11 because Pyo supports this version.
Download [Python 3.11](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe) and install it with activated PATH in the installer.

If you have multiple python version you can set your desired version to the active one. You can also just define this in the virtual environment, but here the information in case you want to set it.
- Open the command line, see the python version you have:
```
py --list
```
- Select 3.11:
```
py -3.11
```

## Download senseSpace
- Go to the folder where you like to install senseSpace
- Open the command line and type:
```
git clone https://github.com/xohm/senseSpace.git
```

## Setup the Environment
- Go into the senseSpace folder
- Setup the Python virtual environment:

**Command Prompt:**
```
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

**PowerShell:**
```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Check Version**
```
python --verison
```
- this needs to output the 3.11 version

## Install Libraries
- Be sure you are in the senseSpace root directory:

**Command Prompt + PowerShell:**
```
pip install -e .
```

- Install the senseSpace library, go to "senseSpace\libs\senseSpaceLib"

**Command Prompt + PowerShell:**
```
cd .\libs\senseSpaceLib
pip install -e .
```

### GPU Acceleration
In case you have a Nvidia GPU, setup the Cuda supported path, this accelerates all the AI responses.
- Setup the Cuda PyTorch version, newer GPU should use Cuda 12.1:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- For older Nvidia GPU:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

To verify CUDA is working:
```
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

This should output `CUDA available: True` if properly installed.

## Install Examples

### Speech
- go to "senseSpace\client\examples\speech"
```
cd .\client\examples\speech
pip install -r requirements.txt
```
- This examples should download all needed data, so you only have to start it. But you have to check if your microphone or speaker device is set, since the defaults might not work. You can check which devices you have very simple, just start the example and check the output:
```
python speech_audio_io.py
```
- The output should be something like this
```
======================================================================
🎤 INPUT devices:
  [ 0] Microsoft Sound Mapper - Input
  [ 1] Microphone (2- HD Pro Webcam C9
  [ 2] Microphone (Realtek(R) Audio)
  [ 3] Headset Microphone (Oculus Virt
  [ 4] Echo Cancelling Speakerphone (J
  [10] Primary Sound Capture Driver

🔊 OUTPUT devices:
  [ 5] Microsoft Sound Mapper - Output
  [ 6] Lautsprecher (Realtek(R) Audio)
  [ 7] DELL U4021QW (HD Audio Driver f
  [ 8] Echo Cancelling Speakerphone (J
  [ 9] Headphones (Oculus Virtual Audi
  [15] Primary Sound Driver
  [16] Lautsprecher (Realtek(R) Audio)
  [17] DELL U4021QW (HD Audio Driver for Display Audio)
  [18] Echo Cancelling Speakerphone (Jabra SPEAK 510 USB)
  [19] Headphones (Oculus Virtual Audio Device)
```
- Select your input and output accordingly:
```
python speech_audio_io.py  --mic 0 --speaker 5
```
- Be carefull with setting the devices, you might get errors if you set this wrong:
```
[AUDIO] Using specified output device 5: Microsoft Sound Mapper - Output
❌ Error in VAD recording: Unanticipated host error [PaErrorCode -9999]: 'There is no driver installed on your system.' [MME error 6]
```