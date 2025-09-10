# senseSpace

SenseSpace is a distributed, interactive media art environment designed to gather, track, and transform human interactions. Within this container, the presence and movement of visitors are sensed and translated into a shared spatial experience. The work explores how bodies, data, and space intersect, creating an evolving choreography of participation where technology and perception merge.


# Install Client

## Linux
Install the Micromamba:
> "${SHELL}" <(curl -L micro.mamba.pm/install.sh)

## Mac
Install the Micromamba:
> brew install micromamba

## Windows
> Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1 -UseBasicParsing).Content)

# Install Server
The server needs a Cuda GPU to run. 

Install ZED SDK
https://www.stereolabs.com/en-ch/developers/release/5.0#82af3640d775
This insalls all you need to run for the Zed camera as well as the pyzed library.

# Setup Environment

Create environment:
> micromamba create -f environment.yml

Activate
> micromamba activate senseSpace

