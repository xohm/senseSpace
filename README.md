# senseSpace

SenseSpace is a distributed, interactive media art environment designed to gather, track, and transform human interactions. Within this container, the presence and movement of visitors are sensed and translated into a shared spatial experience. The work explores how bodies, data, and space intersect, creating an evolving choreography of participation where technology and perception merge.

# Overview
SenseSpace consists of a server and a client.
The server connects to a ZED stereo camera and distributes people’s movement data over the network to connected clients. It must run on a system with an NVIDIA GPU, as it relies on CUDA for processing.
The clients, on the other hand, run with standard Python and only require basic libraries.

Both the server and client require Python 3.12.

# Installtion

If you do not run the server then you only have to setup the client.

## Windows
- Download Python from the server and install it: https://www.python.org/downloads/windows/
- Check that is installed included in your PATH, there is a checkbox in the installer.

## Mac
```
brew update
brew install python@3.12
```

# Setup
Move into the projec folder, setup and active the Python environment.:
```
cd senseSpace
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```
Install the dependencies:
```
pip install -r requirements.txt
```
Now you are in the Python environment for this project. If you start your computer again you need to activate the environment again:
```
source .venv/bin/activate
```

## Client
If there server runs you can start the client. The Ip address and the port is from the running server. You need to insert this from your server settings. 
```
python3 senseSpaceClient.py --viz --server 192.168.1.100 --port 12346  
```

If you like to use the client without any visualization then start it without the --viz parameter:
```
python3 senseSpaceClient.py --server 192.168.1.100 --port 12346  
```

## Server

The server needs a CUDA computer and the ZED Sterocameras connected. First you need to install the ZED SDK and the ZED Python SDK.

### Install ZED SDK
- Download the ZED SDK from: https://www.stereolabs.com/en-ch/developers/release
- Run the installer.
- Go to the ZED installation folder:
```
cd /usr/local/zed
python3 get_python_api.py
```
### Run the Server
To run the server you need to activate the python environment if it is not already activated. You need to be in the main project folder:
```
source .venv/bin/activate
```
Then go into the server folder and start the server(with visualization):
```
cd server
python3 senseSpace_fusion_main.py  --viz
```

## Install Visual Studio Code
[Go to Setup Page](./setup.md)