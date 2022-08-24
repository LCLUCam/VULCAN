# Vulcan Framework

## Introduction

[VULCAN](https://github.com/exoclime/VULCAN) is a photochemical kinetics integrator, created by Shang-Min (Shami) Tsai, for exoplanetary atmospheres written entirely using Python. VULCAN is implemented with the equilibrium chemistry code [FastChem](https://github.com/exoclime/FastChem), created by Daniel Kitzmann and Joachim Stock, which can be used to initialise a state in chemical equilibrium.

The Vulcan Framework within SCUM preserves the scientific methodology of VULCAN while extending its capabilities to 3D simulation and coupling its lower boundary to [Mistra](https://github.com/Mistra-UEA/Mistra), a 1D physical and chemical model of the marine boundary layer (MBL). Moreover, VULCAN is also integrated with the perturbations module, which enables direct modification of the chemical composition of the atmopshere.

## Modus Operandi

### File Structure

```
├── vulcan_framework/
│   ├── atm/
│   ├── fastchem_vulcan/
│   ├── plot_py/
│   ├── thermo/
│   │   ├── NASA9/
│   │   ├── photo_cross/
│   │   ├── Test-networks/
│   │   ├── all_compose.txt
│   │   ├── gibbs_text.txt
│   │   ├── NCHO_photo_network.txt
│   │   ├── ... other thermo & photo networks
│   ├── tools/
│   ├── build_atm.py
│   ├── chem_funs.py
│   ├── make_chem_funs.py
│   ├── op.py
│   ├── phy_const.py
│   ├── plot_vul.py
│   ├── ReadmeVulcan.md
│   ├── runVulcan.py
│   ├── store.py
│   ├── vulcan_cfg.py
│   ├── vulcanController.py
```

### Concept

A **Vulcan run** is initiated by creating an instance of the **vulcanController** class, as instructed by **masterController**, which sits outside the Vulcan Framework. In line with the original working principle of VULCAN, the program is still configured by modifying and importing `vulcan_cfg.py`, in true pythonic fashion. However, in the new Vulcan Framework, in order to accomodate further modifications and parameters from external modules, the user does not edit `vulcan_cfg.py` directly.

Instead, the user has high level control over the Vulcan Framework through the following files:
```
├── input/
    ├── globalParameters.json
    ├── VulcanModelParameters.txt
    ├── VulcanCfgComparator.txt
    ├── VulcanPopulator.json
    ├── ...
```

`globalParameters.json` : shared with other modules, controls world dimensions

`VulcanBaseParameters.txt` : tunes scientific basis and methodology, inherited from VULCAN

`VulcanCfgComparator.txt` : compares past and present `vulcan_cfg.py` files to reduce VULCAN run time by reusing or modifying existing output data

`VulcanPopulator.json` : initializes 3D atmosphere with different location-dependent initial conditions

Each **Vulcan run** is subdivided into $`m\times n`$ **Vulcan columns** providing spatial resolution in the horizontal directions. Each column is numerically propagated forwards in time, as determined by **masterController**, and the data is passed on the other modules upon request. Output data files are stored in the `vul-runtime/` sub-folder. Before each subsequent Vulcan run, data passed back from other modules through **masterController** will be used to modify the baseline initial conditions generated by the default VULCAN model. These modifications manifest themselves in the `...modify_atm.json` file that is automatically created in the `output/` folder, and is read by `runVulcan.py`.


```
├── output/
    ├── vul-output/
    ├── vul-runtime/
        ├── vul-log.txt/
        ├── vul-run-1-200-cfgFile.txt
        ├── vul-run-1-200-output.vul
        ├── vul-run-1-200-plot.eps
        ├── vul-run-1-200-plot.png
        ├── ...
        ├── vul-run-2-221-modify_atm.json


```

Naming convention: `[out_name]-run-[run number]-[column ID]-[file type].[file ext]`
- `[out_name]` : as specified in `VulcanBaseParameters.txt`
- `[run number]` : based on number of times a vulcanController instance was created
- `[column ID]` : three digit number, the first is $2$ by default specific to VULCAN, and the next two are orthogonal axes on the horizontal plane
- `[file ext]` : either `.vul`, `.json`, `.txt`

`vul-output/` : output files pertaining to VULCAN at the end of SCUM run

`vul-runtime/` : output files in between vulcan runs, while SCUM is still running

`vul-log.txt` : log file to keep track of all major vulcan events

`vul-run-1-200-cfgFile.txt` : configuration file, copied from `vulcan_cfg.py`

`vul-run-1-200-output.vul` : runtime output file

`vul-run-1-200-plot.eps` : plot file

`vul-run-1-200-plot.png` : plot file

`vul-run-2-221-modify_atm.json` : for subsequent vulcan runs, where data from external modules are incorporated into the starting conditions of a new vulcan run