#!/usr/bin/env python

import jax.numpy as jnp
from jax.scipy import optimize
import numpy as np
from optparse import OptionParser
from jax import grad, jacobian
import matplotlib.pyplot as plt
import glob, os, json
import sys

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

# python3 JaxOptimizer.py --filesLim 1 --odir test

eta_binning = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
et_binning  = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 256]

#######################################################################
## DEFINING LOSS FUNCTION
#######################################################################

def LossFunction(ietas_index, ihad_index, ihad, iem, jets, ietas_rate_index, ihad_rate_index, ihad_rate, iem_rate, SFs):
    
    l_eta = len(eta_binning)
    l_et = len(et_binning)
    SFs = SFs.reshape(l_eta,l_et)

    jet_energies = jets[:,3]
    l1_jet_energies = jnp.zeros_like(jet_energies)

    ihad_flat = ihad.flatten()
    ietas_index_flat = ietas_index.flatten()
    ihad_index_flat = ihad_index.flatten()
    SF_for_these_towers_flat = SFs[ietas_index_flat, ihad_index_flat]

    ihad_rate_flat = ihad_rate.flatten()
    ietas_rate_index_flat = ietas_rate_index.flatten()
    ihad_rate_index_flat = ihad_rate_index.flatten()
    SF_for_these_rate_flat = SFs[ietas_rate_index_flat, ihad_rate_index_flat]        

    # [FIXME] This should be rounded down with int
    ihad_calib_flat = jnp.multiply(ihad_flat, SF_for_these_towers_flat)
    ihad_calib = ihad_calib_flat.reshape(len(ihad_index),81)
    l1_jet_energies = jnp.sum(ihad_calib[:], axis=1)
    l1_jet_em_energies = jnp.sum(iem[:], axis=1)

    #rate stuff
    #calib rate
    ihad_rate_calib_flat = jnp.multiply(ihad_rate_flat, SF_for_these_rate_flat)
    ihad_rate_calib = ihad_rate_calib_flat.reshape(len(ihad_rate_index),81)
    l1_rate_energies = jnp.sum(ihad_rate_calib[:], axis=1)
    l1_rate_em_energies = jnp.sum(iem_rate[:], axis=1)
    l1_rate_sum_energies = (l1_rate_energies + l1_rate_em_energies)/2. #GeV
    
    #uncalib rate
    ihad_rate_uncalib_flat = ihad_rate_flat
    ihad_rate_uncalib = ihad_rate_uncalib_flat.reshape(len(ihad_rate_index),81)        
    l1_rate_energies_uncalib = jnp.sum(ihad_rate_uncalib[:], axis=1)        
    l1_rate_sum_energies_uncalib = (l1_rate_energies_uncalib + l1_rate_em_energies)/2. #GeV        
    full_rate_uncalib = jnp.sum(l1_rate_sum_energies_uncalib)
    rate_uncalib = jnp.sum(l1_rate_sum_energies_uncalib >= 40.)

    binning = jnp.linspace(0,200,201)
    rate_calib = jnp.sum(l1_rate_sum_energies[:, None] > binning[:-1], axis=0)
    threshold_new = jnp.argmax(rate_calib<=rate_uncalib)
    #print("rate_uncalib =",rate_uncalib)
    #print("rate_calib =",rate_calib)
    #print("threshold_new =",threshold_new)

    #compute acceptance
    #ACC = (l1_jet_energies + l1_jet_em_energies)/2.>threshold_new
    #ACC = ACC.astype(int)
    #ACC = ACC.astype(float)
    #print("acceptance =",ACC)
    
    #compute distance
    d = jnp.maximum(0, -(l1_jet_energies + l1_jet_em_energies)/2. + threshold_new)
    
    DIFF = jnp.abs((l1_jet_energies + l1_jet_em_energies) - jet_energies)
    MAPE = jnp.divide(DIFF, jet_energies)
    STD = jnp.std(MAPE)
    MAPE_s = jnp.sum(MAPE)
    #print(MAPE_s)
    return MAPE_s
    #return ACC
    #d = jnp.sum(d)
    #print("d=",d)
    return d

def ProvideData(options, eta_binning, et_binning):

    #######################################################################
    ## READING INPUT
    #######################################################################

    indir = '/data_CMS/cms/motta/CaloL1calibraton/2023_12_13_NtuplesV56/Input2/JetMET_PuppiJet_BarrelEndcap_Pt30_HoTot70/GoodNtuples/tensors'

    list_towers_files = glob.glob(indir + "/towers_*_0.npz")
    list_jets_files = glob.glob(indir + "/jets_*_0.npz")

    towers = jnp.load(list_towers_files[0], allow_pickle=True)['arr_0']
    jets = jnp.load(list_jets_files[0], allow_pickle=True)['arr_0']

    indir_rate = '/data_CMS/cms/motta/CaloL1calibraton/2023_12_13_NtuplesV56/EphemeralZeroBias_BarrelEndcap_Pt30To1000/EphemeralZeroBias0__Run2022G-v1__RAW__GT130XdataRun3Promptv3_CaloParams2023v02_noL1Calib_data/tensors/'
    list_rate_files = glob.glob(indir_rate + "/towers_*_0.npz")    

    list_towers = []
    list_jets = []
    training_stat = 0

    # Limiting the number of files
    if options["filesLim"]:
        for ifile in range(0, options["filesLim"]):
            # print("Reading file {}".format(ifile))
            x = jnp.load(list_towers_files[ifile], allow_pickle=True)['arr_0']
            y = jnp.load(list_jets_files[ifile], allow_pickle=True)['arr_0']
            list_towers.append(x)
            list_jets.append(y)
            training_stat += len(y)

    # Limiting the number of jets
    elif options["jetsLim"]:
        training_stat = 0
        for ifile in range(0, len(list_towers_files)):
            # print("Reading file {}".format(ifile))
            x = jnp.load(list_towers_files[ifile], allow_pickle=True)['arr_0']
            y = jnp.load(list_jets_files[ifile], allow_pickle=True)['arr_0']
            if training_stat + len(y) > options["jetsLim"]:
                stop = options["jetsLim"] - training_stat
                list_towers.append(x[:stop])
                list_jets.append(y[:stop])
                break
            else:
                list_towers.append(x)
                list_jets.append(y)
                training_stat += len(y)
    
    # No limitation
    else:
        for ifile in range(0, len(list_towers_files)):
            print("Reading file {}".format(ifile))
            x = jnp.load(list_towers_files[ifile], allow_pickle=True)['arr_0']
            y = jnp.load(list_jets_files[ifile], allow_pickle=True)['arr_0']
            list_towers.append(x)
            list_jets.append(y)
            training_stat += len(y)

    filesLim_rate = 30
    list_rate = []
            
    for ifile in range(0, len(list_rate_files)):
        if ifile == filesLim_rate: break
        x = jnp.load(list_rate_files[ifile], allow_pickle=True)['arr_0']
        list_rate.append(x)

    towers = jnp.concatenate(list_towers)
    jets = jnp.concatenate(list_jets)
    rate = jnp.concatenate(list_rate)    

    print(" ### INFO: Training on {} jets".format(len(jets)))

    #######################################################################
    ## INITIALIZING SCALE FACTORS
    #######################################################################

    print(" ### INFO: number of eta bins =",len(eta_binning))
    print(" ### INFO: number of et  bins =",len(et_binning))

    SFs = jnp.ones(shape=(len(eta_binning),len(et_binning)))
    # Apply ZS to ieta <= 15 and iet == 1
    eta_binning = jnp.array(eta_binning)
    et_binning = jnp.array(et_binning)
    SFs = jnp.where((eta_binning[:, None] <= 15) & (et_binning[None, :] == 1), 0, SFs)
    SFs_flat = SFs.ravel()
    print(" ### INFO: Zero Suppression applied to ieta <= 15, et == 1")

    print(" ### INFO: Eta binning = ", eta_binning)
    print(" ### INFO: Energy binning = ", et_binning)

    ietas = jnp.argmax(towers[:, :, 3: ], axis=2) + 1
    ietas_index = jnp.argmax(towers[:, :, 3: ], axis=2)

    ihad = towers[:, :, 1]
    iem = towers[:, :, 0]
    ihad_index = np.digitize(ihad, et_binning)

    ietas_rate = jnp.argmax(rate[:, :, 3: ], axis=2) + 1
    ietas_rate_index = jnp.argmax(rate[:, :, 3: ], axis=2)
    ihad_rate = rate[:, :, 1]
    iem_rate = rate[:, :, 0]
    ihad_rate_index = np.digitize(ihad_rate, et_binning)
    
    return ietas_index, ihad_index, ihad, iem, jets, ietas_rate_index, ihad_rate_index, ihad_rate, iem_rate, SFs_flat

def VisualizeLoss(ietas_index, ihad_index, ihad, iem, jets, ietas_rate_index, ihad_rate_index, ihad_rate, iem_rate, SFs_flat):

    nb_sf=1240
    vect1=np.random.random_sample(nb_sf)*2+0.5
    vect2=np.random.random_sample(nb_sf)*2+0.5

    def f(a,b):
        return a*LossFunction(ietas_index[0:100], ihad_index[0:100], ihad[0:100], iem[0:100], jets[0:100], ietas_rate_index, ihad_rate_index, ihad_rate, iem_rate, vect1)+b*LossFunction(ietas_index[0:100], ihad_index[0:100], ihad[0:100], iem[0:100], jets[0:100], ietas_rate_index, ihad_rate_index, ihad_rate, iem_rate, vect2)

    X=[]
    Y=[]
    Z=[]
    
    x=np.linspace(0,1,30)
    y=np.linspace(0,1,30)
    X,Y=np.meshgrid(x,y)
    print(X)
    print(Y)
    Z=f(X,Y)
    print(Z)
    
    fig=plt.figure()
    ax=plt.axes(projection="3d")
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap="viridis",edgecolor="none")
    ax.set_title("surface")
    ax.set_xlabel("x")
    ax.set_xlabel("y")
    ax.set_xlabel("loss")
    plt.show()
    plt.savefig("vis_loss.png")


if __name__ == "__main__" :

    parser = OptionParser()
    parser.add_option("--odir",                   dest="odir",                   default="./",                         help="Output tag of the output folder")
    parser.add_option("--v",                      dest="v",                      default="HCAL",                       help="Calibration target (ECAL, HCAL)")
    parser.add_option("--jetsLim",                dest="jetsLim",                default=None,       type=int,         help="Fix the total amount of jets to be used")
    parser.add_option("--filesLim",               dest="filesLim",               default=None,       type=int,         help="Maximum number of npz files to use")
    parser.add_option("--bs",                     dest="bs",                     default=1,          type=int,         help="Batch size")
    parser.add_option("--lr",                     dest="lr",                     default=0.001,      type=float,       help="Learning rate")
    parser.add_option("--ep",                     dest="ep",                     default=5,          type=int,         help="Number of epochs")
    parser.add_option("--mask",                   dest="mask",                   default=False,   action='store_true', help="Mask low energy SFs")
    (options, args) = parser.parse_args()
    print(options)

    odir = options.odir
    os.system('mkdir -p '+ odir)

    # test = LossFunction(ietas_index, ihad_index, ihad, iem, jets, SFs_flat)
    # print(test)

    #######################################################################
    ## TRAINING
    #######################################################################

    nb_epochs = options.ep
    bs = options.bs
    lvals = []
    dvals = []
    lr = options.lr

    ietas_index, ihad_index, ihad, iem, jets, ietas_rate_index, ihad_rate_index, ihad_rate, iem_rate, SFs_flat = ProvideData(vars(options), eta_binning, et_binning)

    TrainingInfo = {}
    TrainingInfo["LossType"] = "DIFF"
    TrainingInfo["NJets"] = len(jets)
    TrainingInfo["NEpochs"] = nb_epochs
    TrainingInfo["BS"] = bs
    TrainingInfo["LR"] = lr

    LossHistory = {}
    history_dir = odir + '/History'
    os.system("mkdir -p {}".format(history_dir))


    min_energy = np.min(et_binning)
    max_energy = np.max(et_binning)
    energy_step = 1

    head_text = 'energy bins iEt       = [0'
    for i in et_binning: head_text = head_text + ' ,{}'.format(i)
    head_text = head_text + "]\n"

    head_text = head_text + 'energy bins GeV       = [0'
    for i in et_binning: head_text = head_text + ' ,{}'.format(i/2)
    head_text = head_text + "]\n"

    head_text = head_text + 'energy bins GeV (int) = [0'
    for i in et_binning: head_text = head_text + ' ,{}'.format(int(i/2))
    head_text = head_text + "]\n"

    mask = jnp.ones(shape=(len(eta_binning),len(et_binning)))
    if options.mask:
        mask_energy = 8
        mask = jnp.where(jnp.array(et_binning) <= mask_energy, 0, mask)
        mask = mask.ravel()
        print(" ### INFO: Masking applied to et < {}".format(mask_energy))

    print(" ### INFO: Start training with LR = {}, EPOCHS = {}".format(lr, nb_epochs))

    jnp.set_printoptions(threshold=sys.maxsize)

    for ep in range(nb_epochs):
        print("\n *** Starting Epoch", ep)
        for i in np.arange(0, len(ihad), bs):
            if i == len(ihad) - 1: break
            # calculate the loss
            jac = jacobian(LossFunction, argnums=9)(ietas_index[i:i+bs], ihad_index[i:i+bs], ihad[i:i+bs], iem[i:i+bs], jets[i:i+bs], ietas_rate_index[:], ihad_rate_index[:], ihad_rate[:], iem_rate[:], SFs_flat)
            # apply derivative
            SFs_flat = SFs_flat - lr*jac*mask
            #print("len(jac)",len(jac))
            #print("jac =",jac)
            #print("non-nul parameters",jnp.sum(abs(jac) > 0.000001))
            #for i in range(0,len(jac)):
            #    print(jac[i])
            # print loss for each batch
            loss_value = float(np.mean(LossFunction(ietas_index[i:i+bs], ihad_index[i:i+bs], ihad[i:i+bs], iem[i:i+bs], jets[i:i+bs], ietas_rate_index[:], ihad_rate_index[:], ihad_rate[:], iem_rate[:], SFs_flat)))
            if i%10 == 0: print("Looped over {} jets: Loss = {:.4f}".format(i, loss_value))
        # save loss history
        LossHistory[ep] = float(np.mean(LossFunction(ietas_index, ihad_index, ihad, iem, jets, ietas_rate_index, ihad_rate_index, ihad_rate, iem_rate, SFs_flat)))
        SFs = SFs_flat.reshape(len(eta_binning),len(et_binning))
        SFs_inv = np.transpose(SFs)
        SFOutFile = history_dir+'/ScaleFactors_{}_{}.csv'.format(options.v, ep)
        np.savetxt(SFOutFile, SFs_inv, delimiter=",", newline=',\n', header=head_text, fmt=','.join(['%1.4f']*len(eta_binning)))
        # fill 2D histogram with number of jets for each et-eta bin

    SFs = SFs_flat.reshape(len(eta_binning),len(et_binning))
    SFs_inv = np.transpose(SFs)

    SFOutFile = odir + '/ScaleFactors_{}.csv'.format(options.v)
    np.savetxt(SFOutFile, SFs_inv, delimiter=",", newline=',\n', header=head_text, fmt=','.join(['%1.4f']*len(eta_binning)))
    print('\nScale Factors saved to: {}'.format(SFOutFile))
    # jnp.save(options.odir + 'test', SFs)

    TrainingInfo["TrainLoss"] = LossHistory
    json_path = odir + '/training.json'
    json_data = json.dumps(TrainingInfo, indent=2)
    with open(json_path, "w") as json_file:
        json_file.write(json_data)

