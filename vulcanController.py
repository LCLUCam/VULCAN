"""
This script serves as the interface between masterController.py and runVulcan.py, and it essentially contains two classes.

Creating an instance of the vulcanController class in masterController provides the user with two options:
    1. Propagate the upper atmopshere (simulated by Vulcan) forwards in time for a specific duration
    2. Request for the latest data output to be exported to masterController for the other modules

For S/N 1 - propagation, the controller performs the following key tasks:
    1.  sort out directories
    2.  consolidates all inputs,
            - high level parameters as determined by the user, see self.readModelParametersVulcan()
            - fine-tuning of the model as instructed by masterController.py
            - initial conditions passed from other modules
        
        which are saved into a vulcan_cfg.py file (auto generated) that will be read by runVulcan.py when performing numerical
        integration for a certain vulcan column
    3.  after vulcan_cfg.py is prepared, it is checked against previous vulcan_cfg.py files from previous runs for similarities
        to decide whether to skip the run and copy previous output depending on a set of criteria that can be evolved from being identical (present)
    4.  commands invidual vulcan column runs, keeps track etc.
    5.  all the while, key vulcan events are logged in a text file located in output/

For S/N 2 - data export
    1.  extract information from the latest vulcan output for all the columns
    2.  translate to "standard form", as agreed and to be confirmed with the other module requirements
    
"""

# ===== import public libraries ===== #
from doctest import master
import os, shutil, sys, time, subprocess
import json, pickle
import numpy as np
import ast
from types import SimpleNamespace
from enum import Enum


# ===== enum ===== #
class Modules(str, Enum):
    MISTRA = 'MISTRA'
    VULCAN = 'VULCAN'
    PHREEQC = 'PHREEQC'
    PERTURB = 'PERTURB'

class directions(str, Enum):
    IMPORT = 'IMPORT'
    EXPORT = 'EXPORT'

class runType(str, Enum):
    INITIAL_RUN = 'INITIAL_RUN'
    USE_LATEST_VUL = 'USE_LATEST_VUL'
    EXPORT_ONLY = 'EXPORT_ONLY'
    NO_RUN = "NO_RUN"


# ===== vulcan controller class ===== #
class vulcanController:

    def __init__(self, directories, controllerCfgVulcan, importCfgVulcan, vulcanRunType):

        """ creates an instance of Vulcan, one for each run

        Args:
            directories (list of str):      [masterDir, inputDir, outputDir]
                                            masterDir:  directory of the folder (SCUM) holding masterController.py
                                            inputDir:   directory of the input folder
                                            outputDir:  directory of the output folder

            controllerCfgVulcan (dict):     instructions from masterController, used to configure Vulcan

            importCfgVulcan (dict):         data from other modules, e.g., Mistra and perturbation, used to configure Vulcan

            vulcanRunType (str):            'INITIAL_RUN':       first time running Vulcan, virgin Vulcan
                                            'USE_LATEST_VUL':     not the first time, run with existing .vul data as initial conditions
                                            'EXPORT_ONLY':       do not run Vulcan, extract and export latest Vulcan runtime output data for other modules
                                            'NO_RUN':            configure but do not run Vulcan
        """

        # ===== logs CPU time ===== #
        self._controllerCreatedTime = time.time()
        print('\n\nVulcan - Vulcan created')
        
        # ===== directories passed from Controller ===== #
        self._masterDir, self._inputDir, self._outputDir = directories[0], directories[1], directories[2]

        # ===== construct directories and file paths ===== #
        self._globalParametersFilePath = os.path.join(self.inputDir, 'globalParameters.json')
        self._modelParametersFilePath = os.path.join(self.inputDir, 'VulcanBaseParameters-2.txt')
        self._vulcanPopulatorFilePath = os.path.join(self.inputDir, 'VulcanPopulator.json')
        self._vulcanDirFilePath = os.path.join(self.inputDir, 'VulcanDir.json')
        self._cfgComparator = os.path.join(self.inputDir, 'VulcanCfgComparator.txt')

        self._vulcanFrameworkDir = os.path.join(self.masterDir, 'vulcan_framework/')
        self._cfgFilePath = os.path.join(self.vulcanFrameworkDir, 'vulcan_cfg.py')
        self._makeChemFuns = os.path.join(self.vulcanFrameworkDir, 'make_chem_funs.py')

        self._VulcanRuntimeDir = os.path.join(self.outputDir, 'vul-runtime/')
        self._VulcanRuntimeLogFilePath = os.path.join(self.VulcanRuntimeDir, 'vul-log.txt')
        
        # ===== load dictionaries for configuration ===== #
        self._globalParameters = json.load(open(self.globalParametersFilePath))
        self._vulcanDir = json.load(open(self.vulcanDirFilePath))
        self._vulcanPopulator = json.load(open(self.vulcanPopulatorFilePath))
        self._controllerCfgVulcan = controllerCfgVulcan
        self._importCfgVulcan = importCfgVulcan
        
        # ===== export only ===== #
        if vulcanRunType == runType.EXPORT_ONLY.name:
            return
        
        # ===== further configurations ===== #
        self._vulcanRunType = vulcanRunType
        self.currentRunID = '1-'                    # e.g, '2-202', first number refers to number of vulcanControllers created, 202 refers to the column
        self.out_name = ''                          # as defined in VulcanBaseParameters, to be added later
        self.modifyAtmFilePath = ''                 # to be added later, for vulcanRunType = 'useLatestVul' with further modification using data from other modules

        # ===== running Vulcan ===== #
        # initialization functions
        self._modelParametersVulcan = self.readModelParametersVulcan()
        self.makeCfgFile()            
        
        # sorts out output matters
        self.vulcanOutput()

        # execute, or not
        if self.vulcanRunType == runType.NO_RUN.name:
            return
        else:
            self.runFullVulcan()


    def readModelParametersVulcan(self):
        """ reads modelParametersVulcan.txt, essentially high level user configurations

        Returns:
            list of strings:    modelParametersVulcan.txt
        """
        
        with open(self.modelParametersFilePath, 'r') as f:
            lines = f.readlines()
        return lines


    def makeCfgFile(self):
        """ creates vulcan_cfg.py file using information from modelParametersVulcan.txt
            vulcan_cfg.py serves to provide all information needed to run vulcan on its own
            this function alone is not sufficient, further configurations needed, see self.cfgVulcan()

        Returns: null
        """
        if os.path.exists(self.cfgFilePath):
            os.remove(self.cfgFilePath)

        cfgFile = open(self.cfgFilePath, 'w')
        cfgFile.close()
        return


    def vulcanLog(self, type, message):
        """ logs message to vulcan runtime log file in output

        Args:
            type (str):         'a', 'r+', 'w' etc.
            message (str):      the message
        """

        log = open(self.VulcanRuntimeLogFilePath, type)
        log.write(message)
        log.close()
        return


    def vulcanOutput(self):
        """ creates vulcan runtime output folder, and creates vul-log.txt file within
        """

        if self.vulcanRunType == runType.INITIAL_RUN.name:
            if os.path.exists(self.outputDir):
                shutil.rmtree(self.outputDir)
            if os.path.exists(self.VulcanRuntimeDir):
                shutil.rmtree(self.VulcanRuntimeDir)
            os.mkdir(self.outputDir)
            os.mkdir(self.VulcanRuntimeDir)

            # subprocess.call(['chmod', '-R', '+w', self.VulcanRuntimeDir])
            self.vulcanLog('w', f'Virgin Vulcan = False\n\nVulcan controller created: {self.controllerCreatedTime}\nVulcan Python interpreter: {sys.executable}\nVulcan initialized: {time.time()}\n')

        elif not self.vulcanRunType == runType.NO_RUN.name:
            log = open(self.VulcanRuntimeLogFilePath, 'r')
            lines = log.readlines()
            runString = list(filter(lambda x: True if x.split(' ')[0] == 'Run' else False, lines))[-1]
            self.currentRunID = str(int(runString[4]) + 1) + '-'       # 'Run 1-202'
            log.close()
        else:
            pass
        return


    def runFullVulcan(self):
        """ initializes and loops through vulcan columns, for each:
                - configure
                - check for similar past runs
                - run or copy past results
        """

        # ===== data log ===== #
        self.vulcanLog('a+', f'\n\nVulcan full run execute: {time.time()}')
        print('Vulcan - Vulcan full run execute')
        
        # ===== generate columns ===== #
        columnsList = self.generateColumns()

        # ===== loop through all columns ===== #
        for index, column in enumerate(columnsList):

            # update current run ID
            self.currentRunID = self.currentRunID.split('-')[0] + f'-{column}'
            self.vulcanLog('a+', f'\nVulcan column run: {self.currentRunID}')
            print(f'Vulcan - Vulcan column run: {self.currentRunID}')
            
            # configure column
            self.cfgVulcan(index, column)       # index tells you how many columns have been run
            CfgFilePath = os.path.join(self.VulcanRuntimeDir, f'{self.out_name}-run-{self.currentRunID}-cfgFile.txt')
            self.vulcanLog('a+', f'\nVulcan configured: {time.time()}\nRun {self.currentRunID}\nCfgFilePath: {CfgFilePath}\n')
            print(f'Vulcan - Vulcan column configured, {self.currentRunID}')
            
            # check differences with existing saved cfg files
            checkDupResult = self.checkDup()

            if checkDupResult == False:
                
                # ensure .py extension
                base = os.path.splitext(self.cfgFilePath)[0]
                os.rename(self.cfgFilePath, base + '.py')

                # run the column
                runVulcanPath = os.path.join(self.vulcanFrameworkDir, 'runVulcan.py')
                self.vulcanLog('a+', f'Vulcan column execute: {time.time()}')
                print(f'Vulcan - Vulcan column execute, {self.currentRunID}')
                os.system(sys.executable + f' {runVulcanPath}')

            else:

                # copy configuration file
                srcCfg = checkDupResult[1]
                dstCfg = os.path.join(self.VulcanRuntimeDir, f'{self.out_name}-run-{self.currentRunID}-cfgFile.txt')
                shutil.copyfile(srcCfg, dstCfg)

                # identify all other files in vul-runtime related to that file, e.g., output and plots
                # in the future, just code a pointer
                srcRunID = checkDupResult[0].split('-cfgFile.txt')[0].split('-run-')[-1]
                prevOutputFilePathsAll = [os.path.join(self.VulcanRuntimeDir, file) for file in os.listdir(self.VulcanRuntimeDir)]

                prevOutputFilePathsFiltered_1 = list(filter(lambda x: True if srcRunID in x else False, prevOutputFilePathsAll) )
                prevOutputFilePathsFiltered_2 = list(filter(lambda x: False if '-cfgFile.txt' in x else True, prevOutputFilePathsFiltered_1))

                for outputFilePath in prevOutputFilePathsFiltered_2:
                    dst = outputFilePath.replace(srcRunID, self.currentRunID)
                    src = outputFilePath
                    
                    # copy all related output files except the cfg file
                    shutil.copyfile(src, dst)

                self.vulcanLog('a+', f'\nVulcan column duplicate, srcRunID: {srcRunID}')
                print(f'Vulcan - Vulcan column duplicate, srcRunID: {srcRunID}')

            # data logging after the run
            self.vulcanLog('a+', f'\nVulcan column complete: {time.time()}\nRun {self.currentRunID}')

        return


    def cfgVulcan(self, index, column):
        """ create and writes the vulcan_cfg.py file:
                S/N 1   VulcanBaseParameters.txt
                S/N 2   globalParameters.json
                S/N 3   vulcan directories:     i.e., dictionary of directories from VulcanDir.json from input
                S/N 4   controllerCfgVulcan:    information from vulcanController and masterController
                S/N 5   importCfgVulcan:        information from other modules, e.g., from Mistra
                                                if vulcanRuntype = INITIAL_RUN: read VulcanPopulator.json

            Args:
                index (int):            tells you how many columns have been run within the vulcan run, 0-indexed
                column (str):           column ID

        """

        # ===== S/N 1 ===== #
        cfgFile = open(self.cfgFilePath, 'w')
        cfgFile.writelines(self.modelParametersVulcan)

        # ===== S/N 2 ===== #
        cfgFile.write('\n\n\n# ================== global parameters - auto generated ================== #')
        for mother in self.globalParameters.keys():
            for daughter in self.globalParameters[mother].keys():
                cfgFile.write(f'\n{mother}_{daughter} = {self.globalParameters[mother][daughter]}')

        # ===== S/N 3 ===== #
        cfgFile.write('\n\n\n# ================== directories - auto generated ================== #')
        cfgFile.write(f"""\nmaster_dir = '{self.masterDir}'\ninput_dir = '{self.inputDir}'\noutput_dir = '{self.outputDir}'\nvulcan_framework_dir = '{self.vulcanFrameworkDir}'""")
        
        for mother in self.vulcanDir.keys():
            for daughter in self.vulcanDir[mother].keys():
                if mother == 'vulcan_framework':
                    path = os.path.join(self.vulcanFrameworkDir, self.vulcanDir[mother][daughter])
                    cfgFile.write(f"""\n{daughter} = '{path}'""")

                elif mother == 'output':
                    path = os.path.join(self.outputDir, self.vulcanDir[mother][daughter])
                    cfgFile.write(f"""\n{daughter} = '{path}'""")
                
                elif mother == 'input':
                    path = os.path.join(self.inputDir, self.vulcanDir[mother][daughter])
                    cfgFile.write(f"""\n{daughter} = '{path}'""")

        cfgFile.write(f"""\nvulcan_runtime_dir = '{self.VulcanRuntimeDir}'\nvulcan_runtime_log = '{self.VulcanRuntimeLogFilePath}'""")

        cfgFile.write('\n\n\n# ================== vulcan column properties - auto generated ================== #')
        cfgFile.write(f"""\nrun_type ='{self.vulcanRunType}'\nrun_num = '{self.currentRunID}'""")
        cfgFile.close()

        # ===== S/N 4 - 5 ===== #
        cfgFile = open(self.cfgFilePath, 'r')
        cfgData = cfgFile.readlines()

        # obtain out name
        self.out_name = list(filter(lambda line: True if line.split("=")[0].strip() == 'out_name' else False, cfgData))[0].split("=")[1].strip()[1:-1]
        
        cfgFile.close()

        combineDict = self.controllerCfgVulcan

        if self.vulcanRunType == runType.USE_LATEST_VUL.name:
            # at this point, assume we are just using previous column output data without changing anything (from other modules)
            combineDict['ini_mix'] = """'vulcan_ini'"""
            combineDict['atm_type'] = """'vulcan_ini'"""
            
            # add file path of previous output file to reference off
            previousRunID = f'{int(self.currentRunID[0]) - 1}-{self.currentRunID[-3:]}'
            previousRunFilePath = os.path.join(self.VulcanRuntimeDir, f'{self.out_name}-run-{previousRunID}-output.vul')
            combineDict['vul_ini_file_path'] = f"'{previousRunFilePath}'"

            # pick the correct column, store the update dictionary in the vul-runtime output folder as a file named modify_atm.json
            for positionString, updateDict in self.importCfgVulcan.items():
                translatePositionString = self.translateCol(directions.IMPORT.name, positionString)

                if translatePositionString == column:

                    self.modifyAtmFilePath = os.path.join(self.VulcanRuntimeDir, f'{self.out_name}-run-{self.currentRunID}-modify_atm.json')
                    
                    # perform translation, directions.IMPORT.name, when importing into VULCAN
                    for key, value in updateDict.items():
                        if "Height" in key:
                            newValue = self.translateHeight(directions.IMPORT.name, value)
                        elif "Pressure" in key:
                            newValue = self.translateHeight(directions.IMPORT.name, value)
                        updateDict[key] = newValue

                    file = open(self.modifyAtmFilePath, 'w')
                    json.dump(updateDict, file, indent=4)

                    # dumped = json.dumps(updateDict, cls=NumpyEncoder)
                    # with open(self.modifyAtmFilePath, 'w') as f:
                    # json.dump(dumped, f, indent=4)

                    # change to use previous column output data WITH further modifications from Mistra / Perturbations
                    combineDict['ini_mix'] = """'vulcan_ini_modify'"""
                    combineDict['atm_type'] = """'vulcan_ini_modify'"""

        elif self.vulcanRunType == runType.INITIAL_RUN.name:
            for positionString, updateDict in self.vulcanPopulator.items():
                for pos in positionString.replace(' ','').split(','):
                    if pos == column[1:]:
                        combineDict = {**self.controllerCfgVulcan, **updateDict}

        # ===== further modifucations ===== #
        # if not the first column run for a particular full Vulcan Run, no need to remake chem funs
        if index != 0:
            combineDict['remake_chem_funs'] = False

        # mistraVulcanBoundary sets P_b
        combineDict['P_b'] = self.globalParameters['mistraVulcan']['boundPress']

        # updates vulcan_cfg.py file by replace some variables
        cfgFile = open(self.cfgFilePath, 'w')
        for varName, newVar in combineDict.items():
            for count, line in enumerate(cfgData):
                if line.split("=")[0].strip() == varName:
                    cfgData[count] = f'{varName} = {newVar}\n'
        
        # write to cfgFile and close
        cfgFile.writelines(cfgData)
        cfgFile.close()

        return


    def checkDup(self):
        """ for now, returns the file path of an existing cfg file that is exactly identical, this can be relaxed (in output)
            in the future, add further analysis here to reduce run time by using existing data

            Returns:
                boolean: False          
                
                - or -
                
                list of strings:        [previousCfgFilePath, latestCfgFilePath]
                    previousCfgFilePath:    file path of a previous vulcan_cfg file, stored in output/vul-runtime/ that matches the vulcan_cfg.py
                    latestCfgFilePath:      file path of vulcan_cfg.py, stored in controllers/vulcan_framework/vulcan_cfg.py

        """

        # if there is a modify_atm.json file waiting, return False
        if os.path.exists(self.modifyAtmFilePath):
            return False

        # generate list of variables that are not to be used for the comparison
        with open(self.cfgComparator) as f:
            ignoreVarList =  [x.strip() for x in f.readlines()]

        # check log file for all past "CfgFilePath"
        log = open(self.VulcanRuntimeLogFilePath, 'r')
        logLines = log.readlines()
        cfgFilePathAll = list(filter(lambda x: True if x.split(":")[0] == 'CfgFilePath' else False, logLines))
        
        # latest cfg file will be last in list, we are comparing this with all the earlier entries
        if len(cfgFilePathAll) == 1:
            return False
       
        previousCfgFilePaths = [x.split('CfgFilePath: ')[-1].strip() for x in cfgFilePathAll][:-1]
        latestCfgFilePath = os.path.join(self.vulcanFrameworkDir, 'vulcan_cfg.py')          # this is under vulcan_framework
        
        with open(latestCfgFilePath) as f:
            latestCfgFileVarDict = self.checkDupLinesToDict(f.readlines(), ignoreVarList)
        
        # if checkDupLinesToDict fails, quit
        if not latestCfgFileVarDict:
            return False

        # loop through previousCfgFilePaths
        for previousCfgFilePath in previousCfgFilePaths:
            statusFlag = True

            # prepare previousCfgFileVarDict
            with open (previousCfgFilePath) as f:
                previousCfgFileVarDict = self.checkDupLinesToDict(f.readlines(), ignoreVarList)
            
            # if checkDupLinesToDict fails, quit
            if not latestCfgFileVarDict:
                return False
            
            # checking criteria: keys must be the same
            if set(previousCfgFileVarDict.keys()) != set(latestCfgFileVarDict.keys()):
                statusFlag = False
            else:
                for key in previousCfgFileVarDict.keys():
                    if previousCfgFileVarDict[key] != latestCfgFileVarDict[key]:
                        statusFlag = False
                        break
                
            if statusFlag:
                return [previousCfgFilePath, latestCfgFilePath]
        
        return False


    def checkDupLinesToDict(self, cfgFileLines, ignoreVarList):
        """ convert lines of cfg .txt / .py file to a dictionary for easy comparison
            .eval() used to convert strings to appropriate python data type

        Args:
            cfgFileLines (list of str)
            ignoreVarList (list of str):    list of variable names to ignore in the comparison,
                                            e.g., something trivial like run_num 

        Returns:
            dict
        """

        # within each line, remove whitespaces
        lines_1 = [x.strip() for x in cfgFileLines]
        
        # within each line, remove anything after #
        lines_2 = [x.split('#')[0] for x in lines_1]

        # removes lines that are empty
        lines_3 = list(filter(None, lines_2))

        # remove lines starting with #
        lines_4 = list(filter(lambda x: True if x[0] != '#' else False, lines_3))

        dict = {}
        # split by equal sign, convert literal representation to appropriate data type
        for line in lines_4:
            key, value = line.split('=')[0].strip(), line.split('=')[1].strip()
            try:
                newValue = eval(value)
            except:
                return False            
            dict[key] = newValue

        # remove unwanted variables
        for ignoreVar in ignoreVarList:
            dict.pop(ignoreVar, None)

        return dict


    def generateColumns(self):
        """ generates all the columns required in a Vulcan run based on world dimensions as stored in globalParameters.json

        Returns:
            list of strings
        """

        n = SimpleNamespace(**self.globalParameters)

        columns = []
        for x in range(n.world['atmosphereDim'][0]):
            for y in range(n.world['atmosphereDim'][0]):
                columns.append(f'2{x}{y}')

        return columns


    def nearestColumns(self, column):
        """ takes in a Vulcan column and returns a list of physically plausible nearest neighbours columns,
            left right up down - max. 4 for a central column,
            3 for a column at the side
            2 for a column at the corner

        Args:
            column (string):    e.g., 212

        Returns:
            list of strings:    e.g., [201, 211, 222]
        """

        # string
        nearestList = [ f'{int(column[0])}{int(column[1]) + 1}{int(column[2])}',
                        f'{int(column[0])}{int(column[1]) - 1}{int(column[2])}',
                        f'{int(column[0])}{int(column[1])}{int(column[2]) - 1}',
                        f'{int(column[0])}{int(column[1])}{int(column[2]) + 1}'   ]
        
        # removes negative values
        nearestList = list(filter(lambda x: False if '-' in x else True, nearestList))

        return nearestList


    def vulcanExport(self, user, *args):
        """ exports latest vulcan runtime data output (all columns within the latest run) for external modules

        Args:
            user (str):     'mistra' or 'perturbations'
            *args:          further variables depending on the needs of the module

        Returns:
            dict of dict:   format TBC with other modules, for now, e.g.,

                            {   '200': 
                                    {'upperHeight': 5e5, 'lowerHeight': 2e5, 'levelTemperature': 300, \
                                        'Number Density': {'OH': 1e+17, 'H2O': 5e+13, ...}

                                    },
                                
                                '211': 
                                    {'upperHeight': 5e5, 'lowerHeight': 2e5, 'levelTemperature': 300, \
                                        'Number Density': {'OH': 1e+17, 'H2O': 5e+13, ...}

                                    }, ... other columns
                            }
        """
        
        # before translation, the package to be passed to other modules via masterController
        package = {}
        
        # look at log file and obtain list of all filepaths of the latest run
        log = open(self.VulcanRuntimeLogFilePath, 'r')
        outputLine = list(filter(lambda x: True if x.split('Path:')[0] == 'OutputFile' else False, log.readlines()))[-1]
        latestOutputRunID = '-'.join(outputLine.split('-output.')[0].split('/')[-1].split('-')[:-1])          # vul-run-1
        log.close()
        
        outputFilePathsAll = [os.path.join(self.VulcanRuntimeDir, file) for file in os.listdir(self.VulcanRuntimeDir)]
        latestRunFilePaths = list(filter(lambda x: True if '-output' in x and f'{latestOutputRunID}-' in x else False, outputFilePathsAll))       # all the columns, string
        latestRunColumns = [path.split('-output.')[0].split(f'{latestOutputRunID}-')[1] for path in latestRunFilePaths]                           # all the columns, string
        latestRunZipped = zip(latestRunColumns, latestRunFilePaths)

        if user == Modules.MISTRA.name:

            # pickle load everything and extract important informtion first: press, temp, height
            rawOutputAll = {}
            for column, path in latestRunZipped:
                rawColumnDict = {}

                handle = open(path, 'rb')
                vul_data = pickle.load(handle)

                heightList = vul_data['atm']['zco']
                rawLowerHeight, rawUpperHeight = heightList[0], heightList[1]
                rawColumnDict['upperHeight'] = self.translateHeight(directions.EXPORT.name, rawUpperHeight)          # divide by 1e2 to get meters
                rawColumnDict['lowerHeight'] = self.translateHeight(directions.EXPORT.name, rawLowerHeight)          # divide by 1e2 to get meters

                pressureList = vul_data['atm']['pico']
                rawLowerPressure, rawUpperPressure = pressureList[0], pressureList[1]                     # in 0.1 Pa
                rawColumnDict['upperPressure'] = self.translatePressure(directions.EXPORT.name, rawUpperPressure)
                rawColumnDict['lowerPressure'] = self.translatePressure(directions.EXPORT.name, rawLowerPressure)

                temperatureList = vul_data['atm']['Tco']
                rawLevelTemperature = temperatureList[0]                                    # Kelvins
                rawColumnDict['levelTemperature'] = rawLevelTemperature

                rawSpDict = {}
                for sp in vul_data['variable']['species']:
                    rawSpDict[sp] = vul_data['variable']['y'][:,vul_data['variable']['species'].index(sp)][0]
                rawColumnDict['Number Density'] = rawSpDict

                rawOutputAll[column] = rawColumnDict

            # loop through each column, find nearest neighbours, average and then translate
            for column in latestRunColumns:
                avgColumnDict = {}
                nearestListColumns = list(set(self.nearestColumns(column)).intersection(latestRunColumns))
                
                avgLowerHeight = np.average([rawOutputAll[col]['lowerHeight'] for col in nearestListColumns])
                avgUpperHeight = np.average([rawOutputAll[col]['upperHeight'] for col in nearestListColumns])

                avgLowerPressure = np.average([rawOutputAll[col]['lowerPressure'] for col in nearestListColumns])
                avgUpperPressure = np.average([rawOutputAll[col]['upperPressure'] for col in nearestListColumns])

                avgLevelTemperature = np.average([rawOutputAll[col]['levelTemperature'] for col in nearestListColumns])

                avgColumnDict['lowerHeight'] = avgLowerHeight
                avgColumnDict['upperHeight'] = avgUpperHeight
                avgColumnDict['lowerPressure'] = avgLowerPressure
                avgColumnDict['upperPressure'] = avgUpperPressure
                avgColumnDict['levelTemperature'] = avgLevelTemperature

                # for species, get the longest key list
                useSpCol = max(nearestListColumns, key=lambda col: len(rawOutputAll[col]['Number Density'].keys()))       # i.e., if there are more species in one column, use that col
                
                avgSpDict = {}
                for sp in rawOutputAll[useSpCol]['Number Density'].keys():
                    avgSpDict[sp] = np.average([rawOutputAll[col]['Number Density'][sp] for col in nearestListColumns])
                avgColumnDict['Number Density'] = avgSpDict
                avgColumnDict['index'] = 0

                package[self.translateCol(directions.EXPORT.name, column)] = avgColumnDict


        elif user == Modules.PERTURB.name:
            # returns heights and pressures (upper and lower bound), and temperature in that level
            # number density of all species at last time step

            columnInterestList = [self.translateCol(directions.IMPORT.name, col) for col in args[0]]            # [ (2,1,1), (2,0,2) ], convert to string
            height = self.translateHeight(directions.IMPORT.name, args[1])                                      # in meters, multiply by 1e2 before putting into VULCAN

            for columnInterest in columnInterestList:
                columnDict = {}
                for column, path in latestRunZipped:
                    if columnInterest == column:
                        handle = open(path, 'rb')
                        vul_data = pickle.load(handle)
                        break
                        
                heightList = vul_data['atm']['zco']
                upperHeight = list(filter(lambda x: True if x > height else False, heightList))[0]
                lowerHeight = heightList[np.where(heightList == upperHeight)[0][0] - 1]
                lowerHeightIndex = np.where(heightList == lowerHeight)[0][0]
                columnDict['upperHeight'] = self.translateHeight(directions.EXPORT.name, upperHeight)
                columnDict['lowerHeight'] = self.translateHeight(directions.EXPORT.name, lowerHeight)

                pressureList = vul_data['atm']['pico']
                lowerPressure = pressureList[lowerHeightIndex]                 
                upperPressure = pressureList[lowerHeightIndex + 1]
                columnDict['upperPressure'] = self.translatePressure(directions.EXPORT.name, upperPressure)
                columnDict['lowerPressure'] = self.translatePressure(directions.EXPORT.name, lowerPressure)

                temperatureList = vul_data['atm']['Tco']
                levelTemperature = temperatureList[lowerHeightIndex]            # Kelvins
                columnDict['levelTemperature'] = levelTemperature

                spDict = {}
                for sp in vul_data['variable']['species']:
                    spDict[sp] = vul_data['variable']['y'][:,vul_data['variable']['species'].index(sp)][lowerHeightIndex]

                columnDict['Number Density'] = spDict
                columnDict['index'] = int(lowerHeightIndex)
                
                package[self.translateCol(directions.EXPORT.name, columnInterest)] = columnDict
        
        return package


    # ===== section on translation ===== #
    def translateCol(self, direction, col):
        # VULCAN has column as string '202'
        # universal standard has column as tuple of integers (2,0,2)

        if direction == directions.EXPORT.name:
            return eval(f"""({','.join(col)})""")
        elif direction == directions.IMPORT.name:
            return ''.join(col)

       
    def translatePressure(self, direction, pressure):
        # VULCAN has pressure in 0.1 Pa

        if direction == directions.EXPORT.name:
            return pressure / 1e1
        if direction == directions.IMPORT.name:
            return pressure * 1e1


    def translateHeight(self, direction, height):
        # VULCAN has height such that you need to divide by 100 to get height in meters

        if direction == directions.EXPORT.name:
                return height / 1e2
        if direction == directions.IMPORT.name:
                return height * 1e2


    def translateSpecies(self, species):
        """
        if not '_' in species:
            return f'{species}(g)'

        elif 
        # species in the form of H2O_l(g), 
        """
        return

    @property
    def controllerCreatedTime(self):
        return self._controllerCreatedTime
    @property
    def masterDir(self):
        return self._masterDir
    @property
    def inputDir(self):
        return self._inputDir
    @property
    def outputDir(self):
        return self._outputDir
    @property
    def globalParametersFilePath(self):
        return self._globalParametersFilePath
    @property
    def modelParametersFilePath(self):
        return self._modelParametersFilePath
    @property
    def vulcanPopulatorFilePath(self):
        return self._vulcanPopulatorFilePath
    @property
    def vulcanDirFilePath(self):
        return self._vulcanDirFilePath
    @property
    def cfgComparator(self):
        return self._cfgComparator
    @property
    def vulcanFrameworkDir(self):
        return self._vulcanFrameworkDir
    @property
    def cfgFilePath(self):
        return self._cfgFilePath
    @property
    def makeChemFuns(self):
        return self._makeChemFuns
    @property
    def VulcanRuntimeDir(self):
        return self._VulcanRuntimeDir
    @property
    def VulcanRuntimeLogFilePath(self):
        return self._VulcanRuntimeLogFilePath
    @property
    def globalParameters(self):
        return self._globalParameters
    @property
    def vulcanDir(self):
        return self._vulcanDir
    @property
    def vulcanPopulator(self):
        return self._vulcanPopulator
    @property
    def controllerCfgVulcan(self):
        return self._controllerCfgVulcan
    @property
    def importCfgVulcan(self):
        return self._importCfgVulcan
    @property
    def modelParametersVulcan(self):
        return self._modelParametersVulcan
    @property
    def vulcanRunType(self):
        return self._vulcanRunType