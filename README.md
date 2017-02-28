# HILC_demo
Codes for Demo version of the [HILC project](http://visual.cs.ucl.ac.uk/pubs/HILC/index.html)

A visual-based interactive GUI Programming by Demonstration system.

1. User demonstrates a desktop task. (run `genLog.py`)

2. The system asks use to clearify some confusing parts, if any.

3. The system generate runable script which can automatically perform the demonstrated task.

Now only work on Windows environment. Tested with Windows7 and Windows10.

##Dependencies
  * `Python2.7`

  * `pyHook-1.5.1-cp27-none-win_amd64`

  * python packages (normally come with [Python Anaconda Installation](https://www.continuum.io/downloads)) `win32con,win32api,win32ui,win32gui,matplotlib`

  * scikit-learn version < 0.18. IF your machine has version >= 0.18 and is using `Anaconda`, the package can be downgrade by running `conda install scikit-learn=0.17.1`.

  * Download pre-trained models (.pickle files) and pairwise potential (.csv file) from [here](https://drive.google.com/drive/folders/0BxWU2fKZbtBYUFdPWk0xSFFvTFU?usp=sharing).

##Basic Usage
### Demonstration phase
1. run `python genLog.py --p=PATH\TO\SAVE\FILE\` to record user demonstration.

2. demonstrate a task.
   
  * Linear Task

  * Looping Task

  * Standby Task

3. when done, press `Shift + Esc` to end the script.

4. run `python preprocessing_sniffer_log.py --p=PATH\TO\SAVE\FILE` to transform the generated log file to our unified input format (each consecitive records have time different 1/30s).

5. run `python transcribe_basicaction.py --p=PATH\TO\SAVE\FILE` to transcibe the log-file to a sequence of basic actions. <br/>
This script needs to load pre-trained model and pairwise potential files to do basic action classification. The pre-trained models and pairwise potential can be download from [here](https://drive.google.com/drive/folders/0BxWU2fKZbtBYUFdPWk0xSFFvTFU?usp=sharing).

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
