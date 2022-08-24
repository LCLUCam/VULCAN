##### =============== import public modules =============== #####
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.legend as lg
import scipy
import scipy.optimize as sop
import time, os, sys, json



##### =============== housekeeping =============== #####
# Limiting the number of threads
os.environ["OMP_NUM_THREADS"] = "1"

# get present directory
vulcan_framework_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(vulcan_framework_dir)

# import VULCAN configuration file
import vulcan_cfg

# re-generate chem_funs
# chem_funs.py must be present in vulcan_framework
if vulcan_cfg.remake_chem_funs:
    print ('Vulcan - Making chem_funs.py')                          # running prepipe to construch chem_funs.py
    log = open(vulcan_cfg.vulcan_runtime_log, 'a+')
    log.write(f'\nVulcan - Making chem_funs.py: {time.time()}')
    os.system(sys.executable + f' {vulcan_cfg.make_chem_funs}')


# import VULCAN modules
import store, build_atm, op
import phy_const, plot_vul, chem_funs

# print all for debuging
np.set_printoptions(threshold=np.inf)

# for testing only
"""
output_file_dir = os.path.join(vulcan_cfg.vulcan_runtime_dir, f'{vulcan_cfg.out_name}-run-{vulcan_cfg.run_num}-output.txt')
outfile = open(output_file_dir, 'w')
outfile.write(f'hello')
outfile.close()
output = op.Output()
output.save_cfg()
"""

# read in basic chemistry data
with open(vulcan_cfg.com_file, 'r') as f:
    columns = f.readline()                                      # reading in the first line
    num_ele = len(columns.split())-2                            # number of elements (-2 for removing "species" and "mass")
type_list = ['int' for i in range(num_ele)]
type_list.insert(0,'U20'); type_list.append('float')
compo = np.genfromtxt(vulcan_cfg.com_file,names=True,dtype=type_list)
compo_row = list(compo['species'])


##### =============== Vulcan setup =============== #####

# create instances: variables and parameters class
data_var = store.Variables()
data_atm = store.AtmData()
data_para = store.Parameters()

# Build Output
data_para.start_times = np.append(data_para.start_times, [time.time()])
output = op.Output()
output.save_cfg()

# Check Vul_ini
if vulcan_cfg.ini_mix == 'vulcan_ini_modify':
    modifyAtmPath = os.path.join(vulcan_cfg.vulcan_runtime_dir, f'{vulcan_cfg.out_name}-run-{vulcan_cfg.run_num}-modify_atm.json')    # get json path
    updateDict = json.load(open(modifyAtmPath))
else:
    updateDict = np.nan


# Build Atmosphere
data_para.start_times = np.append(data_para.start_times, [time.time()])
make_atm = build_atm.Atm()
data_atm = make_atm.f_pico(data_atm)                                                        # construct pico
data_atm =  make_atm.load_TPK(data_atm, updateDict)                                         # construct Tco and Kzz
make_atm.mol_diff(data_atm)                                                                 # Only setting up ms (the species molecular weight) if use_moldiff == False
if vulcan_cfg.use_condense:                                                                 # calculating the saturation pressure
    make_atm.sp_sat(data_atm)

# Build Network
data_para.start_times = np.append(data_para.start_times, [time.time()])
rate = op.ReadRate()                                        # for reading rates
data_var = rate.read_rate(data_var, data_atm)               # read-in network and calculating forward rates
if vulcan_cfg.use_lowT_limit_rates:                         # for low-T rates e.g. Jupiter
    data_var = rate.lim_lowT_rates(data_var, data_atm)
data_var = rate.rev_rate(data_var, data_atm)                # reversing rates
data_var = rate.remove_rate(data_var)                       # removing rates

# Configure Atmosphere
data_para.start_times = np.append(data_para.start_times, [time.time()])
ini_abun = build_atm.InitialAbun()
data_var = ini_abun.ini_y(data_var, data_atm, updateDict)                       # initialing y and ymix (the number density and the mixing ratio of every species)

data_var = ini_abun.ele_sum(data_var)                                           # storing the initial total number of atoms
data_atm = make_atm.f_mu_dz(data_var, data_atm, output)                         # calculating mean molecular weight, dz, and dzi and plotting TP
make_atm.BC_flux(data_atm)                                                      # specify the BC

# Configure Numerical Solver
data_para.start_times = np.append(data_para.start_times, [time.time()])
solver_str = vulcan_cfg.ode_solver
solver = getattr(op, solver_str)()

# Configure Photo Chemistry
data_para.start_times = np.append(data_para.start_times, [time.time()])
if vulcan_cfg.use_photo:
    rate.make_bins_read_cross(data_var, data_atm)
    #rate.read_cross(data_var)
    make_atm.read_sflux(data_var, data_atm)
    
    # computing the optical depth (tau), flux, and the photolisys rates (J) for the first time 
    solver.compute_tau(data_var, data_atm)
    solver.compute_flux(data_var, data_atm)
    solver.compute_J(data_var, data_atm)
    # they will be updated in op.Integration by the assigned frequence
    
    # removing rates
    data_var = rate.remove_rate(data_var)

# Build Integrator
data_para.start_times = np.append(data_para.start_times, [time.time()])
integ = op.Integration(solver, output)
solver.naming_solver(data_para)                             # Assgining the specific solver corresponding to different B.C.s

# vulcan_framework set up
print('Vulcan - set up complete, integration begins')


# ============== Vulcan integration  ============== #
data_para.start_times = np.append(data_para.start_times, [time.time()])
integ(data_var, data_atm, data_para, make_atm)              # time-steping in the while loop until conv() returns True or count > count_max or time runs out

# ============== cleaning up ============== #
# Saving output
data_para.start_times = np.append(data_para.start_times, [time.time()])
output.save_out(data_var, data_atm, data_para)

# Plotting the output
data_para.start_times = np.append(data_para.start_times, [time.time()])
plot_vul.plot()