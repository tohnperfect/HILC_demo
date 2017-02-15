# HILC_demo
Codes for Demo version of the [HILC project](http://visual.cs.ucl.ac.uk/pubs/HILC/index.html)

A visual-based interactive GUI Programming by Demonstration system.

1. User demonstrates a desktop task. (run `genLog.py`)

2. The system asks use to clearify some confusing parts, if any.

3. The system generate runable script which can automatically perform the demonstrated task.

##Dependencies
  * `Python2.7`

  * `pyHook-1.5.1-cp27-none-win_amd64`

  * python packages (normally come with [Python Anaconda](https://www.continuum.io/downloads)) `win32con,win32api,win32ui,win32gui,matplotlib`


##Basic Usage
### Demonstration phase
1. run `genLog.py --p=PATH\TO\SAVE\FILE\`

2. demonstrate a task.
   
  * Linear Task

  * Looping Task

  * Standby Task

3. when done, press `Shift + Esc` to end the script.

## Reference
If you use codes in this Repo, please cite our paper, Bibtex entry:
```
@inproceedings{IntharahHILC2017,
 author = {Intharah, Thanapong and Turmukhambetov, Daniyar and Brostow, Gabriel J.},
 title = {Help, It Looks Confusing: GUI Task Automation Through Demonstration and Follow-up Questions},
 booktitle = {Proceedings of the 22nd International Conference on Intelligent User Interfaces},
 series = {IUI '17},
 year = {2017},
 location = {Limassol, Cyprus},
 publisher = {ACM},
} 
```
