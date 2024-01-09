import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(000000)
import sys, os, sys
import numpy as np
from array import array
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)
from RDF_Functions import *

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

def GetArraysFromHisto(histo):
    X = [] ; Y = [] ; X_err = [] ; Y_err = []
    for ibin in range(0,histo.GetNbinsX()):
        X.append(histo.GetBinLowEdge(ibin+1) + histo.GetBinWidth(ibin+1)/2.)
        Y.append(histo.GetBinContent(ibin+1))
        X_err.append(histo.GetBinWidth(ibin+1)/2.)
        Y_err.append(histo.GetBinError(ibin+1))
    return X,Y,X_err,Y_err

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--indir",     dest="indir",    default=None)
parser.add_option("--tag",       dest="tag",      default='')
parser.add_option("--outdir",    dest="outdir",   default=None)
parser.add_option("--label",     dest="label",    default='')
parser.add_option("--nEvts",     dest="nEvts",    type=int, default=-1)
parser.add_option("--target",    dest="target",   default=None)
parser.add_option("--reco",      dest="reco",     action='store_true', default=False)
parser.add_option("--gen",       dest="gen",      action='store_true', default=False)
parser.add_option("--unpacked",  dest="unpacked", action='store_true', default=False)
parser.add_option("--raw",       dest="raw",      action='store_true', default=False)
parser.add_option("--jetPtcut",  dest="jetPtcut", type=float, default=None)
parser.add_option("--etacut",    dest="etacut",   type=float, default=None)
parser.add_option("--LooseEle",  dest="LooseEle", action='store_true', default=False)
parser.add_option("--PuppiJet",  dest="PuppiJet", action='store_true', default=False)
parser.add_option("--do_HoTot",  dest="do_HoTot", action='store_true', default=False)
parser.add_option("--do_EoTot",  dest="do_EoTot", action='store_true', default=False)
parser.add_option("--plot_only", dest="plot_only",action='store_true', default=False)
parser.add_option("--no_plot",   dest="no_plot",  action='store_true', default=False)
parser.add_option("--norm",      dest="norm",     action='store_true', default=False)
parser.add_option("--HCALcalib", dest="HCALcalib",action='store_true', default=False)
parser.add_option("--ECALcalib", dest="ECALcalib",action='store_true', default=False)
parser.add_option("--caloParam", dest="caloParam",type=str,   default='')
parser.add_option("--no_CD",     dest="no_CD",   action='store_true', default=False)
(options, args) = parser.parse_args()

cmap = plt.get_cmap('Set1')

# get/create folders
indir = "/data_CMS/cms/motta/CaloL1calibraton/L1NTuples/"+options.indir
outdir = "/data_CMS/cms/motta/CaloL1calibraton/"+options.outdir
label = options.label
os.system('mkdir -p '+outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs')
os.system('mkdir -p '+outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs')
os.system('mkdir -p '+outdir+'/PerformancePlots'+options.tag+'/'+label+'/ROOTs')

#defining binning of histogram
if options.target == 'jet':
    ptBins  = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 90, 110, 130, 160, 200, 500]
    etaBins = [0., 0.5, 1.0, 1.305, 1.479, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.191]
    signedEtaBins = [-5.191, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.479, -1.305, -1.0, -0.5, 0., 0.5, 1.0, 1.305, 1.479, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.191]
if options.target == 'ele':
    ptBins  = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 90, 110, 130, 160, 200]
    etaBins = [0., 0.5, 1.0, 1.305, 1.479, 2.0, 2.5, 3.0]
    signedEtaBins = [-3.0, -2.5, -2.0, -1.479, -1.305, -1.0, -0.5, 0., 0.5, 1.0, 1.305, 1.479, 2.0, 2.5, 3.0]
if options.target == 'met':
    ptBins  = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 90, 110, 130, 160, 200, 500]
    etaBins = [0., 5.191]
    signedEtaBins = [-5.191, 0., 5.191]
HoTotBins = [0, 0.4, 0.8, 0.95, 1.0]
EoTotBins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
x_lim_response = (0,3)
res_bins = 240

thresholds = np.arange(8,150+1)
thresholds2plot = [10, 20, 35, 50, 100, 150]

if not options.plot_only:

    print(" ### INFO: Start loading data")

    # define targetTree
    if options.reco:
        if options.target == 'jet': targetTree = ROOT.TChain("l1JetRecoTree/JetRecoTree")
        if options.target == 'ele': targetTree = ROOT.TChain("l1ElectronRecoTree/ElectronRecoTree")
        if options.target == 'met': targetTree = ROOT.TChain("l1JetRecoTree/JetRecoTree")
    if options.gen:
        targetTree = ROOT.TChain("l1GeneratorTree/L1GenTree")
    targetTree.Add(indir+"/Ntuple*.root")

    # define level1Tree
    if options.unpacked: level1TreeName = "l1UpgradeTree/L1UpgradeTree"
    else:                level1TreeName = "l1UpgradeEmuTree/L1UpgradeTree"
    level1Tree = ROOT.TChain(level1TreeName)
    level1Tree.Add(indir+"/Ntuple*.root")
    targetTree.AddFriend(level1Tree, level1TreeName)

    # define towersTree
    towersTreeName = "l1CaloTowerEmuTree/L1CaloTowerTree"
    towersTree = ROOT.TChain(towersTreeName)
    towersTree.Add(indir+"/Ntuple*.root")
    targetTree.AddFriend(towersTree, towersTreeName)

    df = ROOT.RDataFrame(targetTree)

    print(" ### INFO: End loading data")

    nEntries = df.Count().GetValue()
    print(" ### INFO: Total entries", nEntries)

    # run on entries specified by user, or only on entries available if that is exceeded
    nevents = options.nEvts
    if (nevents > nEntries) or (nevents==-1): nevents = nEntries
    df = df.Range(nevents)

    print(" ### INFO: Reading", nevents, "events")
    df = df.Range(nevents)

    ##################################################################
    ##################################################################

    if options.target == 'jet':

        # online
        df = df.Define("L1_n",      "L1Upgrade.nJets")
        df = df.Define("L1_eta",    "L1Upgrade.jetEta")
        df = df.Define("L1_phi",    "L1Upgrade.jetPhi")
        if options.raw: df = df.Define("L1_pt",     "L1Upgrade.jetRawEt / 2")
        else:           df = df.Define("L1_pt",     "L1Upgrade.jetEt")

        # offline
        if options.PuppiJet:
            df = df.Define("Offline_n",     "Jet.puppi_nJets")
            df = df.Define("Offline_pt",    "Jet.puppi_etCorr")
            df = df.Define("Offline_eta",   "Jet.puppi_eta")
            df = df.Define("Offline_phi",   "Jet.puppi_phi")
        else:
            df = df.Define("Offline_n",     "Jet.nJets")
            df = df.Define("Offline_pt",    "Jet.etCorr")
            df = df.Define("Offline_eta",   "Jet.eta")
            df = df.Define("Offline_phi",   "Jet.phi")

    ##################################################################
    ##################################################################
            
    if options.target == 'ele':
        
        # online
        df = df.Define("L1_n",      "L1Upgrade.nEGs")
        df = df.Define("L1_eta",    "L1Upgrade.egEta")
        df = df.Define("L1_phi",    "L1Upgrade.egPhi")
        if options.raw: df = df.Define("L1_pt",     "L1Upgrade.egRawEt / 2")
        else:           df = df.Define("L1_pt",     "L1Upgrade.egEt")

        # offline
        df = df.Define("Offline_n",     "Electron.nElectrons")
        df = df.Define("Offline_pt",    "Electron.et")
        df = df.Define("Offline_eta",   "Electron.eta")
        df = df.Define("Offline_phi",   "Electron.phi")

    ##################################################################
    ##################################################################
        
    if options.target == 'met':

        # online
        df = df.Define("L1_n",      "L1Upgrade.nSums")
        df = df.Define("L1_eta",    0)
        df = df.Define("L1_phi",    0)
        df = df.Define("L1_pt",     "L1Upgrade.sumIEt / 2").Filter("L1Upgrade.sumType == 8")
        
        # offline
        df = df.Define("Offline_n",     1)
        df = df.Define("Offline_eta",   0)
        df = df.Define("Offline_phi",   0)
        if options.PuppiJet:
            df = df.Define("Offline_pt", "Sums.puppi_metNoMu")
        else:
            df = df.Define("Offline_pt", "Sums.pfMetNoMu")
            
    ##################################################################
    ########################### APPLY CUTS ###########################

    # skip jets that cannot be reconstructed by L1 (limit is 5.191)
    cut_pt = -1; cut_eta = 5.0; cut_phi = -1
            
    if options.jetPtcut: 
        cut_pt = float(options.jetPtcut)

    if options.etacut:
        cut_eta = float(options.etacut)

    # [FIXME] Yet to be implemented
    if options.target == 'ele' and options.LooseEle:
        sys.exit(" ERROR: This is not implemented yet")
    #     df = df.Define("isLooseElectron", "Electron.isLooseElectron")

    df = df.Define("Offline_pt_cut", 
            "CutOffline(Offline_pt, Offline_eta, Offline_phi, {}, {}, {}).at(0)".format(cut_pt, cut_eta, cut_phi))
    df = df.Define("Offline_eta_cut", 
            "CutOffline(Offline_pt, Offline_eta, Offline_phi, {}, {}, {}).at(1)".format(cut_pt, cut_eta, cut_phi))
    df = df.Define("Offline_phi_cut", 
            "CutOffline(Offline_pt, Offline_eta, Offline_phi, {}, {}, {}).at(2)".format(cut_pt, cut_eta, cut_phi))
        
    ##################################################################    
    ########################### MATCHING #############################

    df = df.Define("good_L1_id", "Matching(L1_pt, L1_eta, L1_phi, Offline_pt_cut, Offline_eta_cut, Offline_phi_cut).at(0)")
    df = df.Define("good_Of_id", "Matching(L1_pt, L1_eta, L1_phi, Offline_pt_cut, Offline_eta_cut, Offline_phi_cut).at(1)")

    df = df.Filter("(good_L1_id != -1) && (good_Of_id != -1)")
    df = df.Define("good_L1_pt",    "L1_pt.at(good_L1_id)")
    df = df.Define("good_L1_eta",   "L1_eta.at(good_L1_id)")
    df = df.Define("good_L1_phi",   "L1_phi.at(good_L1_id)")
    df = df.Define("good_Of_pt",    "Offline_pt_cut.at(good_Of_id)")
    df = df.Define("good_Of_eta",   "Offline_eta_cut.at(good_Of_id)")
    df = df.Define("good_Of_phi",   "Offline_phi_cut.at(good_Of_id)")

    # Define response for matched jets
    df = df.Define("Response", "good_L1_pt / good_Of_pt")

    ##################################################################    
    ######################### CHUNKY DONUT ###########################

    df = df.Define("TT_ieta", "L1CaloTower.ieta")
    df = df.Define("TT_iphi", "L1CaloTower.iphi")
    df = df.Define("TT_iem",  "L1CaloTower.iem")
    df = df.Define("TT_ihc",  "L1CaloTower.ihad")
    df = df.Define("TT_iet",  "L1CaloTower.iet")
    # Define overall hcalET information, ihad for ieta < 29 and iet for ieta > 29
    df = df.Define("TT_ihad", "SumHCAL (TT_ihc, TT_iet, TT_ieta)")

    df = df.Define("good_L1_ieta", "FindIeta(good_L1_eta)")
    df = df.Define("good_L1_iphi", "FindIphi(good_L1_phi)")

    df = df.Define("CD_iem",  "ChunkyDonutEnergy (good_L1_ieta, good_L1_iphi, TT_ieta, TT_iphi, TT_iem, TT_ihad, TT_iet).at(0)")
    df = df.Define("CD_ihad", "ChunkyDonutEnergy (good_L1_ieta, good_L1_iphi, TT_ieta, TT_iphi, TT_iem, TT_ihad, TT_iet).at(1)")
    df = df.Define("CD_iet",  "ChunkyDonutEnergy (good_L1_ieta, good_L1_iphi, TT_ieta, TT_iphi, TT_iem, TT_ihad, TT_iet).at(2)")
    df = df.Define("CD_iesum", "CD_iem + CD_ihad")

    df = df.Define("HoTot", "CD_ihad/CD_iet")
    df = df.Define("EoTot", "CD_iem/CD_iet")

    # Define response for chunky donuts
    df = df.Define("Response_CD", "(CD_ihad+CD_iem) / good_Of_pt")
    df = df.Define("Ratio", "CD_iet / good_L1_pt")

    response_name = 'Response_CD'

    if options.HCALcalib or options.ECALcalib:
        from RDF_Calibration import *
        caloParams_file = "/data_CMS/cms/vernazza/L1TCalibration/CMSSW_13_1_0_pre4_Fix/CMSSW_13_1_0_pre4/src/CaloL1CalibrationProducer/caloParams/" + options.caloParam
        save_folder = outdir+'/PerformancePlots'+options.tag+'/'+label+'/ROOTs'

        ROOT.load_HCAL_SFs(caloParams_file, save_folder)
        ROOT.load_HF_SFs(caloParams_file, save_folder)
        df = df.Define("TT_ihad_calib", "CalibrateIhad(TT_ieta, TT_ihad, {})".format(str(options.HCALcalib).lower()))
        
        ROOT.load_ECAL_SFs(caloParams_file, save_folder)
        df = df.Define("TT_iem_calib", "CalibrateIem(TT_ieta, TT_iem, {})".format(str(options.ECALcalib).lower()))
        
        df = df.Define("CD_iem_calib", "ChunkyDonutEnergy (good_L1_ieta, good_L1_iphi, TT_ieta, TT_iphi, TT_iem_calib, TT_ihad_calib, TT_iet).at(0)")
        df = df.Define("CD_ihad_calib", "ChunkyDonutEnergy (good_L1_ieta, good_L1_iphi, TT_ieta, TT_iphi, TT_iem_calib, TT_ihad_calib, TT_iet).at(1)")
        df = df.Define("CD_iesum_calib", "CD_iem_calib + CD_ihad_calib")
        df = df.Define("Response_CD_calib", "(CD_iem_calib + CD_ihad_calib) / good_Of_pt")

        response_name = "Response_CD_calib"
    
    if options.no_CD: response_name = 'Response'
    else:
        # [FIXME] understand why sometimes they are different
        df = df.Filter("(CD_iet == good_L1_pt) && (CD_iesum == good_L1_pt)")

    df_b = df.Filter("abs(good_Of_eta) < 1.305")
    df_e = df.Filter("(abs(good_Of_eta) > 1.479)")
    
    ##################################################################    
    ########################### DEBUGGING ############################

    # print("Test iem 1,1 =",ROOT.TestCalibrateIem(1,1))
    # print("Test ihad 1,1 =",ROOT.TestCalibrateIhad(1,1))

    # print(" ### INFO: Plotting")

    # c = ROOT.TCanvas()
    # # df = df.Define("Response_uncalib", "CD_ihad")
    # # df = df.Define("Response_calib", "CD_ihad_calib")
    # histo1 = df.Histo1D("CD_ihad")
    # histo2 = df.Histo1D("CD_ihad_calib")
    # histo1.Draw()
    # histo2.Draw("SAME")
    # c.SaveAs("test_ihad.png")

    # f = ROOT.TFile("/data_CMS/cms/vernazza/L1TCalibration/CMSSW_13_1_0_pre4_Fix/CMSSW_13_1_0_pre4/src/CaloL1CalibrationProducer/L1Plotting/test.root","RECREATE")
    # histo1.Write()
    # histo2.Write()
    # f.Close()

    # c = ROOT.TCanvas()
    # histo1 = df.Histo2D(("Ratio", "", 50, 0, 2, 50, -5, 5), "Ratio", "good_L1_eta")
    # histo1.Draw()
    # c.SaveAs("ratio_l1pt_eta.png")

    #################################################################    
    #################################################################    
    #################################################################    

    #################################################################    
    ########################## HISTOGRAMS ###########################

    print("\n ### INFO: Define energy histograms")

    # INCLUSIVE HISTOGRAMS
    nbins = 100; min = 0; max = 500
    offline_pt = df.Histo1D(("good_Of_pt", "good_Of_pt", nbins, min, max), "good_Of_pt")
    online_pt = df.Histo1D(("good_L1_pt", "good_L1_pt", nbins, min, max), "good_L1_pt")
    CD_iet = df.Histo1D(("CD_iet", "CD_iet", nbins, min, max), "CD_iet")     
    CD_iesum = df.Histo1D(("CD_iesum", "CD_iesum", nbins, min, max), "CD_iesum")
    if options.HCALcalib or options.ECALcalib:
        CD_iet_calib = df.Histo1D(("CD_iesum_calib", "CD_iesum_calib", nbins, min, max), "CD_iesum_calib")

    print(" ### INFO: Define response histograms")

    # INCLUSIVE HISTOGRAMS
    pt_response_ptInclusive = df.Histo1D(("pt_response_ptInclusive", 
        "pt_response_ptInclusive", res_bins, 0, 3), response_name)
    pt_barrel_resp_ptInclusive = df_b.Histo1D(("pt_barrel_resp_ptInclusive",
        "pt_barrel_resp_ptInclusive", res_bins, 0, 3), response_name)
    pt_endcap_resp_ptInclusive = df_e.Histo1D(("pt_endcap_resp_ptInclusive",
        "pt_endcap_resp_ptInclusive", res_bins, 0, 3), response_name) 

    # PT RESPONSE - PT BINS HISTOGRAMS
    response_ptBins = []
    barrel_response_ptBins = []
    endcap_response_ptBins = []
    for i in range(len(ptBins)-1):

        df_PtBin = df.Filter("(good_Of_pt > {}) && (good_Of_pt < {})".format(ptBins[i], ptBins[i+1]))
        name = "pt_resp_ptBin"+str(ptBins[i])+"to"+str(ptBins[i+1])
        response_ptBins.append(df_PtBin.Histo1D((name, name, res_bins, 0, 3), response_name))

        df_barrel_PtBin = df_b.Filter("(good_Of_pt > {}) && (good_Of_pt < {})".format(ptBins[i], ptBins[i+1]))
        name = "pt_barrel_resp_ptBin"+str(ptBins[i])+"to"+str(ptBins[i+1])
        barrel_response_ptBins.append(df_barrel_PtBin.Histo1D((name, name, res_bins, 0, 3), response_name))

        df_endcap_PtBin = df_e.Filter("(good_Of_pt > {}) && (good_Of_pt < {})".format(ptBins[i], ptBins[i+1]))
        name = "pt_endcap_resp_ptBin"+str(ptBins[i])+"to"+str(ptBins[i+1])
        endcap_response_ptBins.append(df_endcap_PtBin.Histo1D((name, name, res_bins, 0, 3), response_name))

    # PT RESPONSE -  ETA BINS HISTIGRAMS
    absEta_response_ptBins = []
    minusEta_response_ptBins = []
    plusEta_response_ptBins = []
    for i in range(len(etaBins)-1):

        df_EtaBin = df.Filter("(abs(good_Of_eta) > {}) && (abs(good_Of_eta) < {})".format(etaBins[i], etaBins[i+1]))
        name = "pt_resp_AbsEtaBin"+str(etaBins[i])+"to"+str(etaBins[i+1])
        absEta_response_ptBins.append(df_EtaBin.Histo1D((name, name, res_bins, 0, 3), response_name))

        df_MinusEtaBin = df.Filter("(good_Of_eta < -{}) && (good_Of_eta > -{})".format(etaBins[i], etaBins[i+1]))
        name = "pt_resp_MinusEtaBin"+str(etaBins[i])+"to"+str(etaBins[i+1])
        minusEta_response_ptBins.append(df_MinusEtaBin.Histo1D((name, name, res_bins, 0, 3), response_name))

        df_PlusEtaBin = df.Filter("(good_Of_eta > {}) && (good_Of_eta < {})".format(etaBins[i], etaBins[i+1]))
        name = "pt_resp_PlusEtaBin"+str(etaBins[i])+"to"+str(etaBins[i+1])
        plusEta_response_ptBins.append(df_PlusEtaBin.Histo1D((name, name, res_bins, 0, 3), response_name))

    # PT RESPONSE -  H/TOT BINS HISTIGRAMS
    if options.do_HoTot:
        response_HoTotBins = []
        for i in range(len(HoTotBins)-1):
            df_HoTotBin = df.Filter("(HoTot > {}) && (HoTot < {})".format(HoTotBins[i], HoTotBins[i+1]))
            name = "pt_resp_HoTotBin"+str(HoTotBins[i])+"to"+str(HoTotBins[i+1])
            response_HoTotBins.append(df_HoTotBin.Histo1D((name, name, res_bins, 0, 3), response_name))

    # PT RESPONSE -  E/TOT BINS HISTIGRAMS
    if options.do_EoTot:
        response_EoTotBins = []
        for i in range(len(EoTotBins)-1):
            df_EoTotBin = df.Filter("(EoTot > {}) && (EoTot < {})".format(EoTotBins[i], EoTotBins[i+1]))
            name = "pt_resp_EoTotBin"+str(EoTotBins[i])+"to"+str(EoTotBins[i+1])
            response_EoTotBins.append(df_EoTotBin.Histo1D((name, name, res_bins, 0, 3), response_name))

    ##################################################################    
    ########################### RESOLUTION ###########################
        
    print(" ### INFO: Compute resolution and scale")

    # make resolution plots
    pt_resol_fctPt = ROOT.TH1F("pt_resol_fctPt","pt_resol_fctPt",len(ptBins)-1, array('f',ptBins))
    pt_resol_barrel_fctPt = ROOT.TH1F("pt_resol_barrel_fctPt","pt_resol_barrel_fctPt",len(ptBins)-1, array('f',ptBins))
    pt_resol_endcap_fctPt = ROOT.TH1F("pt_resol_endcap_fctPt","pt_resol_endcap_fctPt",len(ptBins)-1, array('f',ptBins))
    pt_resol_fctEta = ROOT.TH1F("pt_resol_fctEta","pt_resol_fctEta",len(signedEtaBins)-1, array('f',signedEtaBins))

    pt_scale_fctPt = ROOT.TH1F("pt_scale_fctPt","pt_scale_fctPt",len(ptBins)-1, array('f',ptBins))
    pt_scale_fctEta = ROOT.TH1F("pt_scale_fctEta","pt_scale_fctEta",len(signedEtaBins)-1, array('f',signedEtaBins))

    pt_scale_max_fctPt = ROOT.TH1F("pt_scale_max_fctPt","pt_scale_max_fctPt",len(ptBins)-1, array('f',ptBins))
    pt_scale_max_fctEta = ROOT.TH1F("pt_scale_max_fctEta","pt_scale_max_fctEta",len(signedEtaBins)-1, array('f',signedEtaBins))

    if options.do_HoTot:
        pt_resol_fctHoTot = ROOT.TH1F("pt_resol_fctHoTot","pt_resol_fctHoTot",len(HoTotBins)-1, array('f',HoTotBins))
        pt_scale_fctHoTot = ROOT.TH1F("pt_scale_fctHoTot","pt_scale_fctHoTot",len(HoTotBins)-1, array('f',HoTotBins))
        pt_scale_max_fctHoTot = ROOT.TH1F("pt_scale_max_fctHoTot","pt_scale_max_fctHoTot",len(HoTotBins)-1, array('f',HoTotBins))

    if options.do_EoTot:
        pt_resol_fctEoTot = ROOT.TH1F("pt_resol_fctEoTot","pt_resol_fctEoTot",len(EoTotBins)-1, array('f',EoTotBins))
        pt_scale_fctEoTot = ROOT.TH1F("pt_scale_fctEoTot","pt_scale_fctEoTot",len(EoTotBins)-1, array('f',EoTotBins))
        pt_scale_max_fctEoTot = ROOT.TH1F("pt_scale_max_fctEoTot","pt_scale_max_fctEoTot",len(EoTotBins)-1, array('f',EoTotBins))

    for i in range(len(barrel_response_ptBins)):
        pt_scale_fctPt.SetBinContent(i+1, response_ptBins[i].GetMean())
        pt_scale_fctPt.SetBinError(i+1, response_ptBins[i].GetMeanError())

        pt_scale_max_fctPt.SetBinContent(i+1, response_ptBins[i].GetBinCenter(response_ptBins[i].GetMaximumBin()))
        pt_scale_max_fctPt.SetBinError(i+1, response_ptBins[i].GetBinWidth(response_ptBins[i].GetMaximumBin()))

        if response_ptBins[i].GetMean() > 0:
            pt_resol_fctPt.SetBinContent(i+1, response_ptBins[i].GetRMS()/response_ptBins[i].GetMean())
            pt_resol_fctPt.SetBinError(i+1, response_ptBins[i].GetRMSError()/response_ptBins[i].GetMean())
        else:
            pt_resol_fctPt.SetBinContent(i+1, 0)
            pt_resol_fctPt.SetBinError(i+1, 0)

        if barrel_response_ptBins[i].GetMean() > 0:
            pt_resol_barrel_fctPt.SetBinContent(i+1, barrel_response_ptBins[i].GetRMS()/barrel_response_ptBins[i].GetMean())
            pt_resol_endcap_fctPt.SetBinError(i+1, barrel_response_ptBins[i].GetRMSError()/barrel_response_ptBins[i].GetMean())
        else:
            pt_resol_barrel_fctPt.SetBinContent(i+1, 0)
            pt_resol_endcap_fctPt.SetBinError(i+1, 0)        

        if endcap_response_ptBins[i].GetMean() > 0:
            pt_resol_endcap_fctPt.SetBinContent(i+1, endcap_response_ptBins[i].GetRMS()/endcap_response_ptBins[i].GetMean())
            pt_resol_endcap_fctPt.SetBinError(i+1, endcap_response_ptBins[i].GetRMSError()/endcap_response_ptBins[i].GetMean())
        else:
            pt_resol_endcap_fctPt.SetBinContent(i+1, 0)
            pt_resol_endcap_fctPt.SetBinError(i+1, 0)

    for i in range(len(minusEta_response_ptBins)):
        pt_scale_fctEta.SetBinContent(len(etaBins)-1-i, minusEta_response_ptBins[i].GetMean())
        pt_scale_fctEta.SetBinError(len(etaBins)-1-i, minusEta_response_ptBins[i].GetMeanError())
        pt_scale_fctEta.SetBinContent(i+len(etaBins), plusEta_response_ptBins[i].GetMean())
        pt_scale_fctEta.SetBinError(i+len(etaBins), plusEta_response_ptBins[i].GetMeanError())

        pt_scale_max_fctEta.SetBinContent(len(etaBins)-1-i, minusEta_response_ptBins[i].GetBinCenter(minusEta_response_ptBins[i].GetMaximumBin()))
        pt_scale_max_fctEta.SetBinError(len(etaBins)-1-i, minusEta_response_ptBins[i].GetBinWidth(minusEta_response_ptBins[i].GetMaximumBin()))
        pt_scale_max_fctEta.SetBinContent(i+len(etaBins), plusEta_response_ptBins[i].GetBinCenter(plusEta_response_ptBins[i].GetMaximumBin()))
        pt_scale_max_fctEta.SetBinError(i+len(etaBins), plusEta_response_ptBins[i].GetBinWidth(plusEta_response_ptBins[i].GetMaximumBin()))

        if minusEta_response_ptBins[i].GetMean() > 0:
            pt_resol_fctEta.SetBinContent(len(etaBins)-1-i, minusEta_response_ptBins[i].GetRMS()/minusEta_response_ptBins[i].GetMean())
            pt_resol_fctEta.SetBinError(len(etaBins)-1-i, minusEta_response_ptBins[i].GetRMSError()/minusEta_response_ptBins[i].GetMean())
        else:
            pt_resol_fctEta.SetBinContent(len(etaBins)-1-i, 0)
            pt_resol_fctEta.SetBinError(len(etaBins)-1-i, 0)

        if plusEta_response_ptBins[i].GetMean() > 0:
            pt_resol_fctEta.SetBinContent(i+len(etaBins), plusEta_response_ptBins[i].GetRMS()/plusEta_response_ptBins[i].GetMean())
            pt_resol_fctEta.SetBinError(i+len(etaBins), plusEta_response_ptBins[i].GetRMSError()/plusEta_response_ptBins[i].GetMean())
        else:
            pt_resol_fctEta.SetBinContent(i+len(etaBins), 0)
            pt_resol_fctEta.SetBinError(i+len(etaBins), 0)

    if options.do_HoTot:
        for i in range(len(HoTotBins)-1):
            pt_scale_fctHoTot.SetBinContent(i+1, response_HoTotBins[i].GetMean())
            pt_scale_fctHoTot.SetBinError(i+1, response_HoTotBins[i].GetMeanError())
            pt_scale_max_fctHoTot.SetBinContent(i+1, response_HoTotBins[i].GetBinCenter(response_HoTotBins[i].GetMaximumBin()))
            pt_scale_max_fctHoTot.SetBinError(i+1, response_HoTotBins[i].GetBinWidth(response_HoTotBins[i].GetMaximumBin()))
            if response_HoTotBins[i].GetMean() > 0:
                pt_resol_fctHoTot.SetBinContent(i+1, response_HoTotBins[i].GetRMS()/response_HoTotBins[i].GetMean())
                pt_resol_fctHoTot.SetBinError(i+1, response_HoTotBins[i].GetRMSError()/response_HoTotBins[i].GetMean())
            else:
                pt_resol_fctHoTot.SetBinContent(i+1, 0)
                pt_resol_fctHoTot.SetBinError(i+1, 0)

    if options.do_EoTot:
        for i in range(len(EoTotBins)-1):
            pt_scale_fctEoTot.SetBinContent(i+1, response_EoTotBins[i].GetMean())
            pt_scale_fctEoTot.SetBinError(i+1, response_EoTotBins[i].GetMeanError())
            pt_scale_max_fctEoTot.SetBinContent(i+1, response_EoTotBins[i].GetBinCenter(response_EoTotBins[i].GetMaximumBin()))
            pt_scale_max_fctEoTot.SetBinError(i+1, response_EoTotBins[i].GetBinWidth(response_EoTotBins[i].GetMaximumBin()))
            if response_EoTotBins[i].GetMean() > 0:
                pt_resol_fctEoTot.SetBinContent(i+1, response_EoTotBins[i].GetRMS()/response_EoTotBins[i].GetMean())
                pt_resol_fctEoTot.SetBinError(i+1, response_EoTotBins[i].GetRMSError()/response_EoTotBins[i].GetMean())
            else:
                pt_resol_fctEoTot.SetBinContent(i+1, 0)
                pt_resol_fctEoTot.SetBinError(i+1, 0)   

    ##################################################################    
    ##################################################################    
    ################################################################## 

    print(" ### INFO: Saving resolution to root format")
    fileout = ROOT.TFile(outdir+'/PerformancePlots'+options.tag+'/'+label+'/ROOTs/resolution_graphs_'+label+'_'+options.target+'.root','RECREATE')
    offline_pt.Write()
    online_pt.Write()
    CD_iet.Write()
    CD_iesum.Write()
    if options.HCALcalib or options.ECALcalib:
        CD_iet_calib.Write()
    pt_response_ptInclusive.Write()
    pt_barrel_resp_ptInclusive.Write()
    pt_endcap_resp_ptInclusive.Write()
    for i in range(len(ptBins)-1):
        response_ptBins[i].Write()
        barrel_response_ptBins[i].Write()
        endcap_response_ptBins[i].Write()
    for i in range(len(etaBins)-1):
        absEta_response_ptBins[i].Write()
        minusEta_response_ptBins[i].Write()
        plusEta_response_ptBins[i].Write()
    pt_scale_fctPt.Write()
    pt_scale_max_fctPt.Write()
    pt_resol_fctPt.Write()
    pt_scale_fctEta.Write()
    pt_scale_max_fctEta.Write()
    pt_resol_fctEta.Write()
    pt_resol_barrel_fctPt.Write()
    pt_resol_endcap_fctPt.Write()
    if options.do_HoTot:
        for i in range(len(response_HoTotBins)):
            response_HoTotBins[i].Write()
        pt_scale_fctHoTot.Write()
        pt_scale_max_fctHoTot.Write()
        pt_resol_fctHoTot.Write()
    if options.do_EoTot:
        for i in range(len(response_EoTotBins)):
            response_EoTotBins[i].Write()
        pt_scale_fctEoTot.Write()
        pt_scale_max_fctEoTot.Write()
        pt_resol_fctEoTot.Write()
    
    fileout.Close()

    ##################################################################    
    ########################### TURN ON CURVES #######################

    print("\n ### INFO: Computing turn ons for thresholds [{}, ... {}]".format(thresholds[0], thresholds[-1]))
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 150, 180, 250]

    CD_iesum_name = "CD_iesum"
    if options.HCALcalib or options.ECALcalib:
        CD_iesum_name = "CD_iesum_calib"

    total = df.Histo1D(("total", "total", len(bins)-1, array('f',bins)), "good_Of_pt")
    df_er2p5 = df.Filter("good_Of_eta < 2.5")
    total_er2p5 = df_er2p5.Histo1D(("total_Er2p5", "total_Er2p5", len(bins)-1, array('f',bins)), "good_Of_pt")
    df_er1p305 = df.Filter("good_Of_eta < 1.305")
    total_er1p305 = df_er1p305.Histo1D(("total_Er1p305", "total_Er1p305", len(bins)-1, array('f',bins)), "good_Of_pt")

    passing = []
    for i, threshold in enumerate(thresholds):
        df_cut = df.Filter("{} > {}".format(CD_iesum_name, threshold))
        name = "passing_"+str(int(threshold))
        passing.append(df_cut.Histo1D((name, name, len(bins)-1, array('f',bins)), "good_Of_pt"))

    passing_er2p5 = []
    for i, threshold in enumerate(thresholds):
        df_er2p5_cut = df_er2p5.Filter("{} > {}".format(CD_iesum_name, threshold))
        name = "passing_Er2p5_"+str(int(threshold))
        passing_er2p5.append(df_er2p5_cut.Histo1D((name, name, len(bins)-1, array('f',bins)), "good_Of_pt"))

    passing_er1p305 = []
    for i, threshold in enumerate(thresholds):
        df_er1p305_cut = df_er1p305.Filter("{} > {}".format(CD_iesum_name, threshold))
        name = "passing_Er1p305_"+str(int(threshold))
        passing_er1p305.append(df_er1p305_cut.Histo1D((name, name, len(bins)-1, array('f',bins)), "good_Of_pt"))

    print(" ### INFO: Saving turn on to root format")
    fileout = ROOT.TFile(outdir+'/PerformancePlots'+options.tag+'/'+label+'/ROOTs/efficiency_histos_'+label+'_'+options.target+'.root','RECREATE')
    total.Write()
    total_er2p5.Write()
    total_er1p305.Write()
    for i, thr in enumerate(thresholds): 
        passing[i].Write()
        passing_er2p5[i].Write()
        passing_er1p305[i].Write()
    fileout.Close()

    filein = ROOT.TFile(outdir+'/PerformancePlots'+options.tag+'/'+label+'/ROOTs/efficiency_histos_'+label+'_'+options.target+'.root')
    total = filein.Get('total')
    total_er2p5 = filein.Get('total_Er2p5')
    total_er1p305 = filein.Get('total_Er1p305')
    passing = []
    turnons = []
    passing_er2p5 = []
    turnons_er2p5 = []
    passing_er1p305 = []
    turnons_er1p305 = []
    for i, thr in enumerate(thresholds): 
        passing.append(filein.Get("passing_"+str(int(thr))))
        turnons.append(ROOT.TGraphAsymmErrors(passing[i], total, "cp"))
        passing_er2p5.append(filein.Get("passing_Er2p5_"+str(int(thr))))
        turnons_er2p5.append(ROOT.TGraphAsymmErrors(passing_er2p5[i], total_er2p5, "cp"))
        passing_er1p305.append(filein.Get("passing_Er1p305_"+str(int(thr))))
        turnons_er1p305.append(ROOT.TGraphAsymmErrors(passing_er1p305[i], total_er1p305, "cp"))
    filein.Close()

    fileout = ROOT.TFile(outdir+'/PerformancePlots'+options.tag+'/'+label+'/ROOTs/efficiency_graphs_'+label+'_'+options.target+'.root','RECREATE')
    for i, thr in enumerate(thresholds): 
        turnons[i].Write()
        turnons_er2p5[i].Write()
        turnons_er1p305[i].Write()
    fileout.Close()

    if options.no_plot:
        sys.exit()

    ############################################################################################
    ############################################################################################
    ############################################################################################

else:
    print(" ### INFO: Read existing root files")
    filein = ROOT.TFile(outdir+'/PerformancePlots'+options.tag+'/'+label+'/ROOTs/resolution_graphs_'+label+'_'+options.target+'.root')
    pt_scale_fctPt = filein.Get('pt_scale_fctPt')
    pt_scale_fctEta = filein.Get('pt_scale_fctEta')
    if options.do_HoTot:
        pt_scale_fctHoTot = filein.Get('pt_scale_fctHoTot')
        pt_scale_max_fctHoTot = filein.Get('pt_scale_max_fctHoTot')
        pt_resol_fctHoTot = filein.Get('pt_resol_fctHoTot')
    if options.do_EoTot:
        pt_scale_fctEoTot = filein.Get('pt_scale_fctEoTot')
        pt_scale_max_fctEoTot = filein.Get('pt_scale_max_fctEoTot')
        pt_resol_fctEoTot = filein.Get('pt_resol_fctEoTot')
    pt_scale_max_fctPt = filein.Get('pt_scale_max_fctPt')
    pt_scale_max_fctEta = filein.Get('pt_scale_max_fctEta')
    pt_resol_fctPt = filein.Get('pt_resol_fctPt')
    pt_resol_barrel_fctPt = filein.Get('pt_resol_barrel_fctPt')
    pt_resol_endcap_fctPt = filein.Get('pt_resol_endcap_fctPt')
    pt_resol_fctEta = filein.Get('pt_resol_fctEta')
    pt_response_ptInclusive = filein.Get('pt_response_ptInclusive')
    pt_barrel_resp_ptInclusive = filein.Get('pt_barrel_resp_ptInclusive')
    pt_endcap_resp_ptInclusive = filein.Get('pt_endcap_resp_ptInclusive')
    response_ptBins = []
    barrel_response_ptBins = []
    endcap_response_ptBins = []
    for i in range(len(ptBins)-1):
        response_ptBins.append(filein.Get("pt_resp_ptBin"+str(ptBins[i])+"to"+str(ptBins[i+1])))
        barrel_response_ptBins.append(filein.Get("pt_barrel_resp_ptBin"+str(ptBins[i])+"to"+str(ptBins[i+1])))
        endcap_response_ptBins.append(filein.Get("pt_endcap_resp_ptBin"+str(ptBins[i])+"to"+str(ptBins[i+1])))
    absEta_response_ptBins = []
    minusEta_response_ptBins = []
    plusEta_response_ptBins = []
    for i in range(len(etaBins)-1):
        absEta_response_ptBins.append(filein.Get("pt_resp_AbsEtaBin"+str(etaBins[i])+"to"+str(etaBins[i+1])))
        minusEta_response_ptBins.append(filein.Get("pt_resp_MinusEtaBin"+str(etaBins[i])+"to"+str(etaBins[i+1])))
        plusEta_response_ptBins.append(filein.Get("pt_resp_PlusEtaBin"+str(etaBins[i])+"to"+str(etaBins[i+1])))
    if options.do_HoTot:
        response_HoTotBins = []
        for i in range(len(HoTotBins)-1):
            response_HoTotBins.append(filein.Get("pt_resp_HoTotBin"+str(HoTotBins[i])+"to"+str(HoTotBins[i+1])))
    if options.do_EoTot:
        response_EoTotBins = []
        for i in range(len(EoTotBins)-1):
            response_EoTotBins.append(filein.Get("pt_resp_EoTotBin"+str(EoTotBins[i])+"to"+str(EoTotBins[i+1])))

if options.norm:
    y_label_response = 'a.u.'
    for i in range(len(response_ptBins)):
        if response_ptBins[i].Integral() > 0:
            response_ptBins[i].Scale(1.0/response_ptBins[i].Integral())
        if barrel_response_ptBins[i].Integral() > 0:
            barrel_response_ptBins[i].Scale(1.0/barrel_response_ptBins[i].Integral())
        if endcap_response_ptBins[i].Integral() > 0:
            endcap_response_ptBins[i].Scale(1.0/endcap_response_ptBins[i].Integral())

    for i in range(len(minusEta_response_ptBins)):
        if minusEta_response_ptBins[i].Integral() > 0:
            minusEta_response_ptBins[i].Scale(1.0/minusEta_response_ptBins[i].Integral())
        if plusEta_response_ptBins[i].Integral() > 0:
            plusEta_response_ptBins[i].Scale(1.0/plusEta_response_ptBins[i].Integral())
        if absEta_response_ptBins[i].Integral() > 0:
            absEta_response_ptBins[i].Scale(1.0/absEta_response_ptBins[i].Integral())

    if options.do_HoTot:
        for i in range(len(response_HoTotBins)):
            if response_HoTotBins[i].Integral() > 0:
                response_HoTotBins[i].Scale(1.0/response_HoTotBins[i].Integral())

    if options.do_EoTot:
        for i in range(len(response_EoTotBins)):
            if response_EoTotBins[i].Integral() > 0:
                response_EoTotBins[i].Scale(1.0/response_EoTotBins[i].Integral())

else:
    y_label_response = 'Entries'

############################################################################################
############################################################################################
############################################################################################

if options.reco:    targ_name = 'offline'
elif options.gen:   targ_name = 'gen'
if options.target == 'jet':     part_name = 'jet'
elif options.target == 'ele':   part_name = 'e'
elif options.target == 'met':   part_name = 'MET'

barrel_label = r'Barrel $|\eta^{%s, %s}|<1.305$' % (part_name, targ_name)
endcap_label = r'Endcap $1.479<|\eta^{%s, %s}|<5.191$' % (part_name, targ_name)
inclusive_label = r'Inclusive $|\eta^{%s, %s}|<5.191$' % (part_name, targ_name)

x_label_pt      = r'$p_{T}^{%s, %s}$' % (part_name, targ_name)
x_label_eta     = r'$\eta^{%s, %s}$' % (part_name, targ_name)
x_label_Hotot   = r'$H/Tot$'
x_label_Eotot   = r'$E/Tot$'

x_lim_pt        = (0,150)
x_lim_eta       = (-5.2,5.2) # (-3.01,3.01)
x_lim_Hotot     = (0,1)
x_lim_Eotot     = (0,1)

legend_label_pt     = r'$<|p_{T}^{%s, %s}|<$' % (part_name, targ_name)
legend_label_eta    = r'$<|\eta^{%s, %s}|<$' % (part_name, targ_name)
legend_label_Hotot  = r'$<H/Tot<$'
legend_label_Eotot  = r'$<E/Tot<$'

x_label_response = r'$E_{T}^{%s, L1} / p_{T}^{%s, %s}$' % (part_name,part_name, targ_name)
y_label_response = 'Entries'

y_label_resolution  = 'Energy resolution'
y_label_scale       = 'Energy scale (Mean)'
y_label_scale_max   = 'Energy scale (Maximum)'
y_lim_scale = (0.5,1.5)

def SetStyle(ax, x_label, y_label, x_lim, y_lim, leg_title=''):
    leg = plt.legend(loc = 'upper right', fontsize=20, title=leg_title, title_fontsize=18)
    leg._legend_box.align = "left"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.grid()
    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    if options.reco: mplhep.cms.label(data=False, rlabel='(13.6 TeV)')
    else:            mplhep.cms.label('Preliminary', data=True, rlabel=r'110 pb$^{-1}$ (13.6 TeV)') ## 110pb-1 is Run 362617

def AddRectangles(ax, Ymax):
    rect1 = patches.Rectangle((-1.479, 0), 0.174, Ymax*1.3, linewidth=1, edgecolor='gray', facecolor='gray', zorder=2)
    rect2 = patches.Rectangle((1.305, 0), 0.174, Ymax*1.3, linewidth=1, edgecolor='gray', facecolor='gray', zorder=2)
    ax.add_patch(rect1)
    ax.add_patch(rect2)  

############################################################################################
print(" ### INFO: Produce plots inclusive")
############################################################################################

############################################################################################
## response inclusive 

fig, ax = plt.subplots(figsize=(10,10))
X,Y,X_err,Y_err = GetArraysFromHisto(pt_barrel_resp_ptInclusive)
ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, label=barrel_label, lw=2, marker='o', color=cmap(0))
Ymax = max(Y)
X,Y,X_err,Y_err = GetArraysFromHisto(pt_endcap_resp_ptInclusive)
ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, label=endcap_label, lw=2, marker='o', color=cmap(1))
Ymax = max(Ymax, max(Y))
X,Y,X_err,Y_err = GetArraysFromHisto(pt_response_ptInclusive)
ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, label="Inclusive", lw=2, marker='o', color=cmap(2))
Ymax = max(Ymax, max(Y))
SetStyle(ax, x_label_response, y_label_response, x_lim_response, (0,1.3*Ymax))
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/response_ptInclusive_'+label+'_'+options.target+'.pdf')
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/response_ptInclusive_'+label+'_'+options.target+'.png')
plt.close()

############################################################################################
print(" ### INFO: Produce plots in pt bins")
############################################################################################

############################################################################################
## response in pt bins

for i in range(len(barrel_response_ptBins)):
    fig, ax = plt.subplots(figsize=(10,10))
    X,Y,X_err,Y_err = GetArraysFromHisto(barrel_response_ptBins[i])
    ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, label=barrel_label, lw=2, marker='o', color=cmap(0))
    Ymax = max(Y)
    X,Y,X_err,Y_err = GetArraysFromHisto(endcap_response_ptBins[i])
    ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, label=endcap_label, lw=2, marker='o', color=cmap(1))
    Ymax = max(Ymax, max(Y))
    SetStyle(ax, x_label_response, y_label_response, x_lim_response, (0,1.3*Ymax), str(ptBins[i])+legend_label_pt+str(ptBins[i+1]))
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/response_'+str(ptBins[i])+"pt"+str(ptBins[i+1])+'_'+label+'_'+options.target+'.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/response_'+str(ptBins[i])+"pt"+str(ptBins[i+1])+'_'+label+'_'+options.target+'.png')
    plt.close()

############################################################################################
## resolution in pt bins

fig, ax = plt.subplots(figsize=(10,10))
X,Y,X_err,Y_err = GetArraysFromHisto(pt_resol_barrel_fctPt)
ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, label=barrel_label, lw=2, marker='o', color=cmap(0))
Ymax = max(Y)
X,Y,X_err,Y_err = GetArraysFromHisto(pt_resol_endcap_fctPt)
ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, label=endcap_label, lw=2, marker='o', color=cmap(1))
Ymax = max(Ymax, max(Y))
SetStyle(ax, x_label_pt, y_label_resolution, x_lim_pt, (0,1.3*Ymax))
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/resolution_ptBins_'+label+'_'+options.target+'.pdf')
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/resolution_ptBins_'+label+'_'+options.target+'.png')
plt.close()

############################################################################################
## scale in pt bins

fig, ax = plt.subplots(figsize=(10,10))
X,Y,X_err,Y_err = GetArraysFromHisto(pt_scale_fctPt)
ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, lw=2, marker='o', color=cmap(0))
SetStyle(ax, x_label_pt, y_label_scale, x_lim_pt, y_lim_scale)
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/scale_ptBins_'+label+'_'+options.target+'.pdf')
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/scale_ptBins_'+label+'_'+options.target+'.png')
plt.close()

############################################################################################
## scale from maximum in pt bins

fig, ax = plt.subplots(figsize=(10,10))
X,Y,X_err,Y_err = GetArraysFromHisto(pt_scale_max_fctPt)
ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, lw=2, marker='o', color=cmap(0))
SetStyle(ax, x_label_pt, y_label_scale, x_lim_pt, y_lim_scale)
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/scale_max_ptBins_'+label+'_'+options.target+'.pdf')
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/scale_max_ptBins_'+label+'_'+options.target+'.png')
plt.close()

############################################################################################
print(" ### INFO: Produce plots in eta bins")
############################################################################################      

############################################################################################
## response in eta bins

for i in range(len(absEta_response_ptBins)):
    fig, ax = plt.subplots(figsize=(10,10))
    X,Y,X_err,Y_err = GetArraysFromHisto(absEta_response_ptBins[i])
    ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, label=str(etaBins[i])+legend_label_eta+str(etaBins[i+1]), lw=2, marker='o', color=cmap(0))
    Ymax = max(Y)
    SetStyle(ax, x_label_response, y_label_response, x_lim_response, (0,1.3*Ymax))
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/response_'+str(etaBins[i])+"eta"+str(etaBins[i+1])+'_'+label+'_'+options.target+'.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/response_'+str(etaBins[i])+"eta"+str(etaBins[i+1])+'_'+label+'_'+options.target+'.png')
    plt.close()

############################################################################################
## resolution in eta bins

fig, ax = plt.subplots(figsize=(10,10))
X,Y,X_err,Y_err = GetArraysFromHisto(pt_resol_fctEta)
ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, ls='None', lw=2, marker='o', color=cmap(0), zorder=1)
Ymax = max(Y)
AddRectangles(ax,Ymax)
SetStyle(ax, x_label_eta, y_label_resolution, x_lim_eta, (0,1.3*Ymax))
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/resolution_etaBins_'+label+'_'+options.target+'.pdf')
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/resolution_etaBins_'+label+'_'+options.target+'.png')
plt.close()

############################################################################################
## scale in eta bins

fig, ax = plt.subplots(figsize=(10,10))
X,Y,X_err,Y_err = GetArraysFromHisto(pt_scale_fctEta)
ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, lw=2, marker='o', color=cmap(0), zorder=1)
AddRectangles(ax,max(Y))
SetStyle(ax, x_label_eta, y_label_scale, x_lim_eta, y_lim_scale)
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/scale_etaBins_'+label+'_'+options.target+'.pdf')
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/scale_etaBins_'+label+'_'+options.target+'.png')
plt.close()

############################################################################################
## scale from maximum in eta bins

fig, ax = plt.subplots(figsize=(10,10))
X,Y,X_err,Y_err = GetArraysFromHisto(pt_scale_max_fctEta)
ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, lw=2, marker='o', color=cmap(0), zorder=1)
AddRectangles(ax,max(Y))
SetStyle(ax, x_label_eta, y_label_scale, x_lim_eta, y_lim_scale)
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/scale_max_etaBins_'+label+'_'+options.target+'.pdf')
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/scale_max_etaBins_'+label+'_'+options.target+'.png')
plt.close()

if options.do_HoTot:

    ############################################################################################
    print(" ### INFO: Produce plots in H/Tot bins")
    ############################################################################################

    ############################################################################################
    ## response in HoTot bins

    fig, ax = plt.subplots(figsize=(10,10))
    Ymax = 0
    for i in range(len(response_HoTotBins)):
        X,Y,X_err,Y_err = GetArraysFromHisto(response_HoTotBins[i])
        ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, label=str(HoTotBins[i])+legend_label_Hotot+str(HoTotBins[i+1]), lw=2, marker='o', color=cmap(i))
        Ymax = max(max(Y), Ymax)        
    SetStyle(ax, x_label_response, y_label_response, x_lim_response, (0,1.3*Ymax))
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/response_HoTot_'+label+'_'+options.target+'.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/response_HoTot_'+label+'_'+options.target+'.png')
    plt.ylim(0.1, Ymax*1.3)
    plt.yscale('log')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/response_HoTot_'+label+'_'+options.target+'_log.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/response_HoTot_'+label+'_'+options.target+'_log.png')
    plt.close()

    ############################################################################################
    ## resolution in HoTot bins

    fig, ax = plt.subplots(figsize=(10,10))
    X,Y,X_err,Y_err = GetArraysFromHisto(pt_resol_fctHoTot)
    ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, ls='None', lw=2, marker='o', color=cmap(0), zorder=1)
    Ymax = max(Y)
    SetStyle(ax, x_label_Hotot, y_label_resolution, x_lim_Hotot, (0,1.3*Ymax))
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/resolution_HoTotBins_'+label+'_'+options.target+'.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/resolution_HoTotBins_'+label+'_'+options.target+'.png')
    plt.close()

    ############################################################################################
    ## scale in HoTot bins

    fig, ax = plt.subplots(figsize=(10,10))
    X,Y,X_err,Y_err = GetArraysFromHisto(pt_scale_fctHoTot)
    ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, lw=2, marker='o', color=cmap(0))
    SetStyle(ax, x_label_Hotot, y_label_scale, x_lim_Hotot, y_lim_scale)
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/scale_HoTotBins_'+label+'_'+options.target+'.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/scale_HoTotBins_'+label+'_'+options.target+'.png')
    plt.close()

    ############################################################################################
    ## scale from maximum in HoTot bins

    fig, ax = plt.subplots(figsize=(10,10))
    X,Y,X_err,Y_err = GetArraysFromHisto(pt_scale_max_fctHoTot)
    ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, lw=2, marker='o', color=cmap(0))
    SetStyle(ax, x_label_Hotot, y_label_scale, x_lim_Hotot, y_lim_scale)
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/scale_max_HoTotBins_'+label+'_'+options.target+'.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/scale_max_HoTotBins_'+label+'_'+options.target+'.png')
    plt.close()

if options.do_EoTot:

    ############################################################################################
    print(" ### INFO: Produce plots in H/Tot bins")
    ############################################################################################

    ############################################################################################
    ## response in EoTot bins

    fig, ax = plt.subplots(figsize=(10,10))
    Ymax = 0
    for i in range(len(response_EoTotBins)):
        X,Y,X_err,Y_err = GetArraysFromHisto(response_EoTotBins[i])
        ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, label=str(EoTotBins[i])+legend_label_Eotot+str(EoTotBins[i+1]), lw=2, marker='o', color=cmap(i))
        Ymax = max(max(Y), Ymax)        
    SetStyle(ax, x_label_response, y_label_response, x_lim_response, (0,1.3*Ymax))
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/response_EoTot'+label+'_'+options.target+'.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/response_EoTot'+label+'_'+options.target+'.png')
    plt.ylim(0.1, Ymax*1.3)
    plt.yscale('log')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/response_EoTot'+label+'_'+options.target+'_log.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/response_EoTot'+label+'_'+options.target+'_log.png')
    plt.close()

    ############################################################################################
    ## resolution in EoTot bins

    fig, ax = plt.subplots(figsize=(10,10))
    X,Y,X_err,Y_err = GetArraysFromHisto(pt_resol_fctEoTot)
    ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, ls='None', lw=2, marker='o', color=cmap(0), zorder=1)
    Ymax = max(Y)
    SetStyle(ax, x_label_Eotot, y_label_resolution, x_lim_Eotot, (0,1.3*Ymax))
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/resolution_EoTotBins_'+label+'_'+options.target+'.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/resolution_EoTotBins_'+label+'_'+options.target+'.png')
    plt.close()

    ############################################################################################
    ## scale in EoTot bins

    fig, ax = plt.subplots(figsize=(10,10))
    X,Y,X_err,Y_err = GetArraysFromHisto(pt_scale_fctEoTot)
    ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, lw=2, marker='o', color=cmap(0))
    SetStyle(ax, x_label_Eotot, y_label_scale, x_lim_Eotot, y_lim_scale)
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/scale_EoTotBins_'+label+'_'+options.target+'.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/scale_EoTotBins_'+label+'_'+options.target+'.png')
    plt.close()

    ############################################################################################
    ## scale from maximum in EoTot bins

    fig, ax = plt.subplots(figsize=(10,10))
    X,Y,X_err,Y_err = GetArraysFromHisto(pt_scale_max_fctEoTot)
    ax.errorbar(X, Y, xerr=X_err, yerr=Y_err, lw=2, marker='o', color=cmap(0))
    SetStyle(ax, x_label_Eotot, y_label_scale, x_lim_Eotot, y_lim_scale)
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/scale_max_EoTotBins_'+label+'_'+options.target+'.pdf')
    plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/scale_max_EoTotBins_'+label+'_'+options.target+'.png')
    plt.close()

############################################################################################
############################################################################################
############################################################################################

############################################################################################
print(" ### INFO: Produce plots turn ons")
############################################################################################

filein = ROOT.TFile(outdir+'/PerformancePlots'+options.tag+'/'+label+'/ROOTs/efficiency_graphs_'+label+'_'+options.target+'.root')
turnons = []
turnons_er2p5 = []
turnons_er1p305 = []
for i, thr in enumerate(thresholds): 
    turnons.append(filein.Get(f'divide_passing_{thr}_by_total'))
    turnons_er2p5.append(filein.Get(f'divide_passing_Er2p5_{thr}_by_total_Er2p5'))
    turnons_er1p305.append(filein.Get(f'divide_passing_Er1p305_{thr}_by_total_Er1p305'))
filein.Close()

if options.reco:
    if options.target == 'jet': x_label = '$E_{T}^{jet, offline}$ [GeV]'
    if options.target == 'ele': x_label = '$E_{T}^{e, offline}$ [GeV]'
    if options.target == 'met': x_label = '$MET_{\mu corrected}^{offline}$ [GeV]'
if options.gen:
    x_label = '$E_{T}^{jet, gen}$ [GeV]'

def SetStyle(ax, x_label):
    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    leg = plt.legend(loc = 'lower right', fontsize=20)
    leg._legend_box.align = "left"
    plt.xlabel(x_label)
    plt.ylabel('Efficiency')
    plt.xlim(0, 220)
    plt.ylim(0, 1.05)
    plt.grid()
    if options.reco: mplhep.cms.label(data=False, rlabel='(13.6 TeV)')
    else:            mplhep.cms.label('Preliminary', data=True, rlabel=r'110 pb$^{-1}$ (13.6 TeV)') ## 110pb-1 is Run 362617

thresholds = list(thresholds)
# cmap = matplotlib.cm.get_cmap('tab20c')
fig, ax = plt.subplots(figsize=(10,10))
for i, thr in enumerate(thresholds2plot):
    X = [] ; Y = [] ; Y_low = [] ; Y_high = []
    turnon = turnons[thresholds.index(thr)]
    for ibin in range(0,turnon.GetN()):
        X.append(turnon.GetPointX(ibin))
        Y.append(turnon.GetPointY(ibin))
        Y_low.append(turnon.GetErrorYlow(ibin))
        Y_high.append(turnon.GetErrorYhigh(ibin))
    ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], label="$p_{T}^{L1} > $"+str(thr)+" GeV", lw=2, marker='o', color=cmap(i))
SetStyle(ax, x_label)
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/turnOns_'+label+'_'+options.target+'.pdf')
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/turnOns_'+label+'_'+options.target+'.png')
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
for i, thr in enumerate(thresholds2plot):
    X = [] ; Y = [] ; Y_low = [] ; Y_high = []
    turnon = turnons_er2p5[thresholds.index(thr)]
    for ibin in range(0,turnon.GetN()):
        X.append(turnon.GetPointX(ibin))
        Y.append(turnon.GetPointY(ibin))
        Y_low.append(turnon.GetErrorYlow(ibin))
        Y_high.append(turnon.GetErrorYhigh(ibin))
    ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], label="$p_{T}^{L1} > $"+str(thr)+" GeV", lw=2, marker='o', color=cmap(i))
SetStyle(ax, x_label)
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/turnOns_Er2p5_'+label+'_'+options.target+'.pdf')
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/turnOns_Er2p5_'+label+'_'+options.target+'.png')
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
for i, thr in enumerate(thresholds2plot):
    X = [] ; Y = [] ; Y_low = [] ; Y_high = []
    turnon = turnons_er1p305[thresholds.index(thr)]
    for ibin in range(0,turnon.GetN()):
        X.append(turnon.GetPointX(ibin))
        Y.append(turnon.GetPointY(ibin))
        Y_low.append(turnon.GetErrorYlow(ibin))
        Y_high.append(turnon.GetErrorYhigh(ibin))
    ax.errorbar(X, Y, xerr=1, yerr=[Y_low, Y_high], label="$p_{T}^{L1} > $"+str(thr)+" GeV", lw=2, marker='o', color=cmap(i))
SetStyle(ax, x_label)
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PDFs/turnOns_Er1p305_'+label+'_'+options.target+'.pdf')
plt.savefig(outdir+'/PerformancePlots'+options.tag+'/'+label+'/PNGs/turnOns_Er1p305_'+label+'_'+options.target+'.png')
plt.close()

