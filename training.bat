cd /d %~dp0\segmentation_unet
call activate.bat u-net
python main.py --gpu --augmentation
pause
