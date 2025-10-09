# senseSpace - Server

The server needs to be calibrated before it can be used. ZED SDK has a tool for this: ZED360.
https://www.stereolabs.com/docs/fusion/zed360

## Room Calibration
- Setup all cameras and connect them to the computer. Be sure to use the right USB plugs, since each camera needs 5Gb/s, select the right plugs. Otherwise the camera will not work.
- Start the calibration software:
´´´
ZED360 
´´´
- In the application select the "Setup your Room" button.
- Set DepthMode to Neural.
- Press "Auto Discover" to get all cameras.
- The press "Setup the Room".
- Press "Start Calibration" and walk through the space which is covered by the cameras.
- Pess "Finish Calibration" when the room is covered and save the calibration JSON file into a folder. Best into the project folder of the server at "senseSpace/server/calib". Now the calibration file is set for this setup.

## Start Server
Go into the server folder and start the server(with visualization):
```
cd server
python3 senseSpace_fusion_main.py  --viz
```
Don't forget to activate the python environment for this.