# phy-contrib

[![Build Status](https://img.shields.io/travis/kwikteam/phy-contrib.svg)](https://travis-ci.org/kwikteam/phy-contrib)
[![PyPI release](https://img.shields.io/pypi/v/phycontrib.svg)](https://pypi.python.org/pypi/phycontrib)

Plugins for [**phy**](https://github.com/kwikteam/phy). Currently, this package provides two integrated spike sorting GUIs:

* **KwikGUI**: to be used with Klusta and the Kwik format (successor of KlustaViewa)
* **TemplateGUI**: to be used with KiloSort and SpykingCircus

## Quick install

You first need to install [**phy**](https://github.com/kwikteam/phy). Then, activate your conda environment and do:

```bash
pip install phycontrib
```

### Installing the development version

for a first pass, just give our github address instead of theirs
BUT, if you want to update it you need to move phy contrib from the github to your site packages 
copy from the github (eg. S:\Vigi\Matlab\GitHub\phy-contrib) to your miniconda version (e.g. C:\Users\User\Miniconda3\envs\phyL\Lib\site-packages\phycontrib) where phyL is your environment

Next, set up your config file
in C:\Users\User\.phy set up two things
1) a plugins folder, with your plugsin
2) a file called phy_config.py
add these lines to it:

c.Plugins.dirs = [r'C:\Users\User\.phy\plugins/']
c.TemplateGUI.plugins = ['ControllerSettings', 'AmplitudeHistogram']


If you want to use the bleeding-edge version, do:

```
git clone https://github.com/kwikteam/phy-contrib.git
cd phy-contrib
python setup.py develop
```

Then, you can update at any time with:

```
cd phy-contrib
git pull
```

## Documentation

* [Main documentation](http://phy-contrib.readthedocs.io/en/latest/)
* KwikGUI documentation (work in progress)
* [TemplateGUI documentation](http://phy-contrib.readthedocs.io/en/latest/template-gui/)
