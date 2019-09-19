Experimental a3c implementation
===============================

Dependecies
-----------
- pytorch >= 1.0 with gpu support
- numpy

Installation
------------
 1.) Put all files in this folder into same folder as flatland
 **OR**
 2.) Add path to flatland in "main.py" to sys.path

Running
-------
* Training: python3 main.py generate
* Replay: python3 main.py test

Settings
--------

**PLEASE NOTE: THIS CODE IS EXPERIMENTAL AND IS AS IT IS!**
**The author make no claim to be complete!**


Files
-----
1.) main.py: this is the entry point for all tests
2.) model.py: here the model is defined
3.) loss.py: the loss used for trainin
4.) observation.py: definition of the state space from flatland
5.) loader.py: pytorch data loader
6.) linker.py: a3c routine
7.) flatland_a3c.ini: paramenter file

Remark on training
------------------
* for sequential training, all paramenters are loaded from flatland_a3c.ini
* for multiprocess training, most of parameters are defined in static functions in linker.py