# HILC_demo
Codes for Demo version of the [HILC project](http://visual.cs.ucl.ac.uk/pubs/HILC/index.html)

A visual-based interactive GUI Programming by Demonstration system.

1. The user demonstrates a desktop task. (see `genLog.py`)

2. The system asks the user to clearify some confusing parts, if any. (see `gen_script_detectors.py`)

3. The system generates runable script which can automatically perform the demonstrated task. (see `script_execution.py`)

Now HILC only works on Windows environment. Tested with Windows7 and Windows10.

##Dependencies
  * `Python2.7`

  * `pyHook-1.5.1` and `pygame-1.9.20`. Both dependencies can be download the `wheel` files [here](https://drive.google.com/drive/folders/0BxWU2fKZbtBYUFdPWk0xSFFvTFU?usp=sharing) and use `pip` to install by running `pip install WHEEL_FILE_NAME`.

  * `pyautogui` for mice and keyboards controlling. the package can be installed by running `pip install pyautogui`

  * python packages (normally come with [Python Anaconda Installation](https://www.continuum.io/downloads)) `win32con,win32api,win32ui,win32gui,matplotlib`

  * scikit-learn version < 0.18. If your machine has version >= 0.18 and is using `Anaconda`, the package can be downgrade by running `conda install scikit-learn=0.17.1`.

  * Download pre-trained models (.pickle files) and pairwise potential (.csv file) from [here](https://drive.google.com/drive/folders/0BxWU2fKZbtBYUFdPWk0xSFFvTFU?usp=sharing).

##Basic Usage

Run the Windows PowerShell script `run_all.ps1 PATH\TO\SAVE\FOLDER\` to run all steps.

###OR

Run each step as follows,

1. [Demonstration phase] run `python genLog.py --p=PATH\TO\SAVE\FOLDER\` to record user demonstration.

2. [Demonstration phase] demonstrate a task.

> * Linear Task : The demonstration procedure for the looping task is simply just a demonstration,

>> Demonstration.

> * Looping Task : for looping tasks users need to input the __Looping signals__ (Ctrl+Shift+L) or (Ctrl+Shift+Break). The demonstration procedure for the looping task is as follows,

>> __Looping Signal__ > Demonstration > __Looping Signal__ > Ctrl+Click on positive instances, providing examples of target patterns > __Looping Signal__.

> * monitoring Task : for monitoring tasks users need to input the __Standby Signals__ (Ctrl+Shift+W) or (Ctrl+Shift+PrtScr). The demonstration procedure for the monitoring task is as follows,

>> When the visual cue appears > __Standby Signal__ > Demonstration.

3. [Demonstration phase] when the task demonstration is done, press `Shift + Esc` to end the sniffing script.

4. [Demonstration phase] run `python preprocessing_sniffer_log.py --p=PATH\TO\SAVE\FOLDER\` to transform the generated log file to our unified input format (each consecitive records have time different 1/30s).

5. [Demonstration phase] run `python transcribe_basicaction.py --p=PATH\TO\SAVE\FOLDER\` to transcibe the log-file to a sequence of basic actions. <br/>
This script needs to load pre-trained model and pairwise potential files to do basic action classification. The pre-trained models and pairwise potential can be download from [here](https://drive.google.com/drive/folders/0BxWU2fKZbtBYUFdPWk0xSFFvTFU?usp=sharing).

6. [Teaching phase] run `python gen_script_detectors.py --p=PATH\TO\SAVE\FOLDER\` the script will train a detector for each basic action. After the detector is trained, the script will test the detector and ask for clarification if needed. In this step, the script needs to load the pre-scripted the questions file `pygame_question_genscript.txt`, which can be download from [here](https://drive.google.com/drive/folders/0BxWU2fKZbtBYUFdPWk0xSFFvTFU?usp=sharing).

7. [Running phase] run `python script_execution.py --p=PATH\TO\SAVE\FOLDER\` to run the generated script.

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
