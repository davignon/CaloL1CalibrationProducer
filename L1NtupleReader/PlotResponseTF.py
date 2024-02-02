import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
import numpy as np
import sys, glob, os
import pandas as pd
import math, warnings
warnings.simplefilter(action='ignore')

import mplhep
plt.style.use(mplhep.style.CMS)

c_uncalib = 'black'
c_oldcalib = 'red'
c_newcalib = 'green'

leg_uncalib = 'No calib'
leg_oldcalib = 'Old calib'
leg_newcalib = 'New calib'

feature_description = {
    'chuncky_donut': tf.io.FixedLenFeature([], tf.string, default_value=''), # byteslist to be read as string 
    'trainingPt'   : tf.io.FixedLenFeature([], tf.float32, default_value=0)  # single float values
}

# parse proto input based on description
def parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    chuncky_donut = tf.io.parse_tensor(example['chuncky_donut'], out_type=tf.float32) # decode byteslist to original 81x43 tensor
    return chuncky_donut, example['trainingPt']

TowersEta = {
    1:  [0,     0.087],    2: [0.087,  0.174],    3: [0.174,  0.261],    4: [0.261,  0.348],    5: [0.348,  0.435],    6: [0.435,  0.522],    7: [0.522,  0.609],    8: [0.609,  0.696],    9: [0.696,  0.783],    10: [0.783,  0.870],
    11: [0.870, 0.957],    12: [0.957, 1.044],    13: [1.044, 1.131],    14: [1.131, 1.218],    15: [1.218, 1.305],    16: [1.305, 1.392],    17: [1.392, 1.479],    18: [1.479, 1.566],    19: [1.566, 1.653],    20: [1.653, 1.740],
    21: [1.740, 1.830],    22: [1.830, 1.930],    23: [1.930, 2.043],    24: [2.043, 2.172],    25: [2.172, 2.322],    26: [2.322, 2.5],      27: [2.5,   2.650],    28: [2.650, 3.],       29: [3., 3.139],       30: [3.139, 3.314],
    31: [3.314, 3.489],    32: [3.489, 3.664],    33: [3.664, 3.839],    34: [3.839, 4.013],    35: [4.013, 4.191],    36: [4.191, 4.363],    37: [4.363, 4.538],    38: [4.538, 4.716],    39: [4.716, 4.889],    40: [4.889, 5.191],}

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def PlotDistribution(df_jets, odir, v_sample, var):

    if var == 'jetPt'   : x_label = r'$p_{T}^{jet, offline} [GeV]$'
    if var == 'jetIEta' : x_label = r'$i\eta^{jet, offline}$'
    fig = plt.figure(figsize = [10,10])
    plt.hist(df_jets[var], bins=100, histtype='step', stacked=True, linewidth=2, color='Blue')
    plt.xlabel(x_label)
    plt.ylabel('Entries')
    plt.grid(linestyle='dotted')
    plt.legend(fontsize=15, loc='upper left')
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
    # plt.title('Jets Resolution {}'.format(v_sample))
    savefile = odir + '/{}_{}.png'.format(var, v_sample)
    plt.savefig(savefile)
    print(savefile)
    plt.close()

def PlotEtaIesum(df_Towers, odir, v_sample, type):
    if v_sample == 'ECAL': x_label = 'iem'
    if v_sample == 'HCAL': x_label = 'hcalET'
    plt.figure(figsize = [10,10])
    plt.title(type)
    sel = df_Towers[x_label] > 0
    max_eta = df_Towers['ieta'].max()
    min_eta = df_Towers['ieta'].min()
    bins_x = np.arange(min_eta,max_eta+1)
    bins_y = np.linspace(0,50,50)
    hist, _, _ = np.histogram2d(df_Towers[sel]['ieta'], df_Towers[sel][x_label], bins=[bins_x,bins_y])
    hist[hist == 0] = -1
    cmap = matplotlib.cm.get_cmap("viridis")
    cmap.set_under(color='white') 
    plt.imshow(hist.T, cmap=cmap, origin='lower', aspect='auto', norm=LogNorm(vmin=0.1, vmax=100000))
    # plt.hist2d(df_Towers[sel]['ieta'], df_Towers[sel][x_label], bins=[bins_x,bins_y], cmap='plasma', vmin=1)
    plt.colorbar()
    plt.xlabel('Eta')
    plt.ylabel(x_label)
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
    # plt.title('Jets Resolution {}'.format(v_sample))
    savefile = odir + '/{}_{}_vs_ieta_{}.png'.format(type, x_label, v_sample)
    plt.savefig(savefile)
    print(savefile)
    plt.close()  

    plt.figure(figsize = [10,10])
    plt.title(type)
    sel = df_Towers[x_label] > 1
    max_eta = df_Towers['ieta'].max()
    min_eta = df_Towers['ieta'].min()
    bins_x = np.arange(min_eta,max_eta+1)
    bins_y = np.linspace(0,50,50)
    hist, _, _ = np.histogram2d(df_Towers[sel]['ieta'], df_Towers[sel][x_label], bins=[bins_x,bins_y])
    hist[hist == 0] = -1
    cmap = matplotlib.cm.get_cmap("viridis")
    cmap.set_under(color='white') 
    plt.imshow(hist.T, cmap=cmap, origin='lower', aspect='auto', norm=LogNorm(vmin=0.1, vmax=20000))
    plt.colorbar()
    plt.xlabel('Eta')
    plt.ylabel(x_label)
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
    # plt.title('Jets Resolution {}'.format(v_sample))
    savefile = odir + '/{}_{}_vs_ieta_{}_No1.png'.format(type, x_label, v_sample)
    plt.savefig(savefile)
    print(savefile)
    plt.close()    

    plt.figure(figsize = [10,10])
    plt.title(type)
    sel = df_Towers[x_label] >= 10
    max_eta = df_Towers['ieta'].max()
    min_eta = df_Towers['ieta'].min()
    bins_x = np.arange(min_eta,max_eta+1)
    bins_y = np.linspace(0,50,50)
    hist, _, _ = np.histogram2d(df_Towers[sel]['ieta'], df_Towers[sel][x_label], bins=[bins_x,bins_y])
    hist[hist == 0] = -1
    cmap = matplotlib.cm.get_cmap("viridis")
    cmap.set_under(color='white') 
    plt.imshow(hist.T, cmap=cmap, origin='lower', aspect='auto', norm=LogNorm(vmin=0.1, vmax=2000))
    plt.colorbar()
    plt.xlabel('Eta')
    plt.ylabel(x_label)
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
    # plt.title('Jets Resolution {}'.format(v_sample))
    savefile = odir + '/{}_{}_vs_ieta_{}_From10.png'.format(type, x_label, v_sample)
    plt.savefig(savefile)
    print(savefile)
    plt.close()    


def PlotRate(df_jets, odir, v_sample):

    cmap = matplotlib.cm.get_cmap('Set1')
    binning = np.linspace(0,200,100)
    b_center = (binning[:-1] + binning[1:])/2

    df_barrel = df_jets[(df_jets['jetIEta'] <= 15)]
    df_endcap = df_jets[(df_jets['jetIEta'] > 15)]
    
    fig = plt.figure(figsize = [10,10])
    e_barrel, _ = np.histogram(df_barrel['jetPtRate'], bins=binning)
    e_endcap, _ = np.histogram(df_endcap['jetPtRate'], bins=binning)
    e_tot, _    = np.histogram(df_jets['jetPtRate'], bins=binning)

    plt.hist(df_barrel['jetPtRate'], bins=binning, histtype='step', stacked=True, linewidth=2, label='Barrel', color=cmap(0))
    plt.hist(df_endcap['jetPtRate'], bins=binning, histtype='step', stacked=True, linewidth=2, label='Endcap', color=cmap(1))
    plt.hist(df_jets['jetPtRate'], bins=binning, histtype='step', stacked=True, linewidth=2, label='Inclusive', color=cmap(2))
    plt.xlabel(r'p_{T}^{jet, L1} [GeV]')
    plt.ylabel('Entries')
    plt.grid(linestyle='dotted')
    plt.legend(fontsize=15, loc='upper left')
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
    # plt.title('Jets Resolution {}'.format(v_sample))
    savefile = odir + '/Rate_PtProgression_{}.png'.format(v_sample)
    plt.savefig(savefile)
    print(savefile)
    plt.yscale("log")
    savefile = odir + '/Rate_PtProgression_{}_log.png'.format(v_sample)
    plt.savefig(savefile)
    print(savefile)
    plt.close()

    rate_barrel = [np.sum(e_barrel[b_center > i])/np.sum(e_tot) for i in b_center]
    rate_endcap = [np.sum(e_endcap[b_center > i])/np.sum(e_tot) for i in b_center]
    rate_tot = [np.sum(e_tot[b_center > i])/np.sum(e_tot) for i in b_center]

    fig = plt.figure(figsize = [10,10])
    plt.plot(b_center, rate_barrel, label='Barrel', marker='o', linestyle='dashed', linewidth=2, color=cmap(0))
    plt.plot(b_center, rate_endcap, label='Endcap', marker='o', linestyle='dashed', linewidth=2, color=cmap(1))
    plt.plot(b_center, rate_tot, label='Inclusive', marker='o', linestyle='dashed', linewidth=2, color=cmap(2))
    plt.xlabel(r'p_{T}^{jet, L1} [GeV]')
    plt.ylabel('Rate Proxy')
    plt.grid(linestyle='dotted')
    plt.legend(fontsize=15, loc='upper left')
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
    # plt.title('Jets Resolution {}'.format(v_sample))
    savefile = odir + '/Rate_Progression_{}.png'.format(v_sample)
    plt.savefig(savefile)
    print(savefile)
    plt.close()

    RateProxy_30 = len(df_jets[(df_jets['jetSeed'] >= 4) & (df_jets['jetPtRate'] > 30)])/len(df_jets)
    print("Rate Proxy target at 30 GeV = {}".format(RateProxy_30))
    RateProxy_40 = len(df_jets[(df_jets['jetSeed'] >= 4) & (df_jets['jetPtRate'] > 40)])/len(df_jets)
    print("Rate Proxy target at 40 GeV = {}".format(RateProxy_40))
    RateProxy_50 = len(df_jets[(df_jets['jetSeed'] >= 4) & (df_jets['jetPtRate'] > 50)])/len(df_jets)
    print("Rate Proxy target at 50 GeV = {}".format(RateProxy_50))

def PlotResolutionInclusive(df_jets, odir, v_sample):

    bins_res = np.linspace(0,3,240)
    cmap = matplotlib.cm.get_cmap('Set1')

    plt.figure(figsize=(10,10))
    sel_barrel = np.abs(df_jets['jetEta']) < 1.305
    text_1 = leg_uncalib+r' barrel : $\mu={:.3f}, res={:.3f}$'.format(df_jets[sel_barrel]['unc_res'].mean(), df_jets[sel_barrel]['unc_res'].std()/df_jets[sel_barrel]['unc_res'].mean())
    plt.hist(df_jets[sel_barrel]['unc_res'], bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(0))
    sel_endcap = np.abs(df_jets['jetEta']) >= 1.305
    text_1 = leg_uncalib+r' endcap : $\mu={:.3f}, res={:.3f}$'.format(df_jets[sel_endcap]['unc_res'].mean(), df_jets[sel_endcap]['unc_res'].std()/df_jets[sel_endcap]['unc_res'].mean())
    plt.hist(df_jets[sel_endcap]['unc_res'], bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(1))
    text_1 = leg_uncalib+r': $\mu={:.3f}, res={:.3f}$'.format(df_jets['unc_res'].mean(), df_jets['unc_res'].std()/df_jets['unc_res'].mean())
    counts, bins, _ = plt.hist(df_jets['unc_res'], bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(2))
    plt.xlabel('Response')
    plt.ylabel('a.u.')
    plt.ylim(0, 1.3*np.max(counts))
    plt.xlim(0,3)
    plt.grid(linestyle='dotted')
    plt.legend(fontsize=15, loc='upper left')
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
    # plt.title('Jets Resolution {}'.format(v_sample))
    savefile = odir + '/Res_{}.png'.format(v_sample)
    plt.savefig(savefile)
    print(savefile)
    plt.close()

    if v_sample == "ECAL":
        plt.figure(figsize=(10,10))
        sel_barrel = np.abs(df_jets['jetEta']) < 1.305
        text_1 = leg_uncalib+r' barrel : $\mu={:.3f}, res={:.3f}$'.format(df_jets[sel_barrel]['unc_res_iem'].mean(), df_jets[sel_barrel]['unc_res_iem'].std()/df_jets[sel_barrel]['unc_res_iem'].mean())
        plt.hist(df_jets[sel_barrel]['unc_res_iem'], bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(0))
        sel_endcap = np.abs(df_jets['jetEta']) >= 1.305
        text_1 = leg_uncalib+r' endcap : $\mu={:.3f}, res={:.3f}$'.format(df_jets[sel_endcap]['unc_res_iem'].mean(), df_jets[sel_endcap]['unc_res_iem'].std()/df_jets[sel_endcap]['unc_res_iem'].mean())
        plt.hist(df_jets[sel_endcap]['unc_res_iem'], bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(1))
        text_1 = leg_uncalib+r': $\mu={:.3f}, res={:.3f}$'.format(df_jets['unc_res_iem'].mean(), df_jets['unc_res_iem'].std()/df_jets['unc_res_iem'].mean())
        counts, bins, _ = plt.hist(df_jets['unc_res_iem'], bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(2))
        plt.xlabel('Response Only Iem')
        plt.ylabel('a.u.')
        plt.ylim(0, 1.3*np.max(counts))
        plt.xlim(0,3)
        plt.grid(linestyle='dotted')
        plt.legend(fontsize=15, loc='upper left')
        mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
        # plt.title('Jets Resolution {}'.format(v_sample))
        savefile = odir + '/Res_{}_Iem.png'.format(v_sample)
        plt.savefig(savefile)
        print(savefile)
        plt.close()

    if v_sample == "HCAL":
        plt.figure(figsize=(10,10))
        sel_barrel = np.abs(df_jets['jetEta']) < 1.305
        text_1 = leg_uncalib+r' barrel : $\mu={:.3f}, res={:.3f}$'.format(df_jets[sel_barrel]['unc_res_ihad'].mean(), df_jets[sel_barrel]['unc_res_ihad'].std()/df_jets[sel_barrel]['unc_res_ihad'].mean())
        plt.hist(df_jets[sel_barrel]['unc_res_ihad'], bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(0))
        sel_endcap = np.abs(df_jets['jetEta']) >= 1.305
        text_1 = leg_uncalib+r' endcap : $\mu={:.3f}, res={:.3f}$'.format(df_jets[sel_endcap]['unc_res_ihad'].mean(), df_jets[sel_endcap]['unc_res_ihad'].std()/df_jets[sel_endcap]['unc_res_ihad'].mean())
        plt.hist(df_jets[sel_endcap]['unc_res_ihad'], bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(1))
        text_1 = leg_uncalib+r': $\mu={:.3f}, res={:.3f}$'.format(df_jets['unc_res_ihad'].mean(), df_jets['unc_res_ihad'].std()/df_jets['unc_res_ihad'].mean())
        counts, bins, _ = plt.hist(df_jets['unc_res_ihad'], bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(2))
        plt.xlabel('Response')
        plt.ylabel('a.u.')
        plt.ylim(0, 1.3*np.max(counts))
        plt.xlim(0,3)
        plt.grid(linestyle='dotted')
        plt.legend(fontsize=15, loc='upper left')
        mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
        # plt.title('Jets Resolution {}'.format(v_sample))
        savefile = odir + '/Res_{}_Ihad.png'.format(v_sample)
        plt.savefig(savefile)
        print(savefile)
        plt.close()

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

def PlotResolutionPtBins(df_jets, odir, v_sample, bin_type):
    
    bins_res = np.linspace(0,3,240)

    if v_sample == 'HCAL':
        if bin_type == 'pt':
            keyBins  = [30, 35, 40, 45, 50, 60, 70, 90, 110, 130, 160, 200, 500]
            key = 'jetPt'
            legend_label = r'$<p_{T}^{jet, offline}<$'
            x_label = r'$p_{T}^{jet, offline}$'
        elif bin_type == 'eta':
            keyBins = [0., 0.5, 1.0, 1.305, 1.479, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.191]
            key = 'jetEta'
            legend_label = r'$<|\eta^{jet, offline}|<$'
            x_label = r'$\eta^{jet, offline}$'
        elif bin_type == 'ieta':
            keyBins = np.arange(1,40)
            key = 'jetIEta'
            legend_label = r'$<|i_{\eta}^{jet, offline}|<$'
            x_label = r'$\eta^{jet, offline}$'
        x_lim = (0.,3.)
        x_label_res = r'$E_{T}^{jet, L1} / p_{T}^{jet, offline}$'
    if v_sample == 'ECAL':
        if bin_type == 'pt':
            keyBins  = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 90, 110, 130, 160, 200]
            key = 'jetPt'
            legend_label = r'$<p_{T}^{e, offline}<$'
            x_label = r'$p_{T}^{e, offline}$'
        elif bin_type == 'eta':
            keyBins = [0., 0.5, 1.0, 1.305, 1.479, 2.0, 2.5, 3.0]
            key = 'jetEta'
            legend_label = r'$<|\eta^{e, offline}|<$'
            x_label = r'$\eta^{e, offline}$'
        x_lim = (0.2,1.5)
        x_label_res = r'$E_{T}^{e/\gamma, L1} / p_{T}^{e, offline}$'

    mean_vs_pt_unc = []
    res_vs_pt_unc = []
    maximum_vs_pt_unc = []

    for i in range(len(keyBins)-1):
        
        Ymax = 0
        fig, ax = plt.subplots(figsize=(10,10))
        if bin_type == 'pt': sel_pt = (df_jets[key] > keyBins[i]*2) & (df_jets[key] < keyBins[i+1]*2)
        elif bin_type == 'eta': sel_pt = (df_jets[key] > keyBins[i]) & (df_jets[key] < keyBins[i+1])
        h = plt.hist(df_jets[sel_pt]['unc_res'], bins=bins_res, label=leg_uncalib, histtype='step', density=True, stacked=True, linewidth=2, color=c_uncalib)
        mean_vs_pt_unc.append(df_jets[sel_pt]['unc_res'].mean())
        res_vs_pt_unc.append(df_jets[sel_pt]['unc_res'].std()/df_jets[sel_pt]['unc_res'].mean())
        if h[0][0] >= 0: maximum_vs_pt_unc.append(h[1][np.where(h[0] == h[0].max())][0])
        else: maximum_vs_pt_unc.append(0)
        if h[0][0] >= 0: Ymax = h[0].max()
        
        for xtick in ax.xaxis.get_major_ticks():
            xtick.set_pad(10)
        leg = plt.legend(loc='upper right', fontsize=20, title=str(keyBins[i])+legend_label+str(keyBins[i+1]), title_fontsize=18)
        leg._legend_box.align = "left"
        plt.xlabel(x_label_res)
        plt.ylabel('a.u.')
        plt.xlim(x_lim)
        plt.ylim(0., Ymax*1.3)
        for xtick in ax.xaxis.get_major_ticks():
            xtick.set_pad(10)
        plt.grid()
        mplhep.cms.label(data=False, rlabel='(13.6 TeV)')
        savefile = odir+'/response_'+str(keyBins[i])+bin_type+str(keyBins[i+1])+'_'+v_sample+'.png'
        plt.savefig(savefile)
        print(savefile)
        plt.close()

    # plot resolution vs keyBins

    fig = plt.figure(figsize = [10,10])
    X = [(keyBins[i] + keyBins[i+1])/2 for i in range(len(keyBins)-1)]
    X_err = [(keyBins[i+1] - keyBins[i])/2 for i in range(len(keyBins)-1)]
    plt.errorbar(X, res_vs_pt_unc, xerr=X_err, label=leg_uncalib, ls='None', lw=2, marker='o', color=c_uncalib, zorder=0)
    Ymax = max(res_vs_pt_unc)

    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    leg = plt.legend(loc='upper right', fontsize=20)
    leg._legend_box.align = "left"
    plt.xlabel(x_label)
    plt.ylabel('Energy resolution')
    plt.xlim(keyBins[0], keyBins[-1])
    try: plt.ylim(0., Ymax*1.3)
    except: pass
    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    plt.grid()
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)')
    savefile = odir+'/resolution_'+bin_type+'Bins_'+v_sample+'.png'
    plt.savefig(savefile)
    print(savefile)
    plt.close()

    # plot scale vs keyBins
    fig = plt.figure(figsize = [10,10])
    plt.errorbar(X, mean_vs_pt_unc, xerr=X_err, label=leg_uncalib, ls='None', lw=2, marker='o', color=c_uncalib, zorder=0)
    Ymax = max(mean_vs_pt_unc)

    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    leg = plt.legend(loc='upper right', fontsize=20)
    leg._legend_box.align = "left"
    plt.xlabel(x_label)
    plt.ylabel('Energy scale')
    plt.xlim(keyBins[0], keyBins[-1])
    try: plt.ylim(0., Ymax*1.3)
    except: pass
    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    plt.grid()
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)')
    savefile = odir+'/scale_'+bin_type+'Bins_'+v_sample+'.png'
    plt.savefig(savefile)
    print(savefile)
    plt.close()

    # plot scale from maximum vs keybins
    fig = plt.figure(figsize = [10,10])
    plt.errorbar(X, maximum_vs_pt_unc, xerr=X_err, label=leg_uncalib, ls='None', lw=2, marker='o', color=c_uncalib, zorder=0)
    Ymax = max(maximum_vs_pt_unc)

    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    leg = plt.legend(loc='upper right', fontsize=20)
    leg._legend_box.align = "left"
    plt.xlabel(x_label)
    plt.ylabel('Energy scale')
    plt.xlim(keyBins[0], keyBins[-1])
    try: plt.ylim(0., Ymax*1.3)
    except: pass
    for xtick in ax.xaxis.get_major_ticks():
        xtick.set_pad(10)
    plt.grid()
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)')
    savefile = odir+'/scale_max_'+bin_type+'Bins_'+v_sample+'.png'
    plt.savefig(savefile)
    print(savefile)
    plt.close()

### To run:
### python3 PlotResponseTF.py --indir 2023_05_01_NtuplesV44 --v HCAL --tag DataReco --filesLim 2 --addtag _A
### python3 PlotResponseTF.py --indir 2023_05_19_NtuplesV46 --v ECAL --tag DataReco --filesLim 2

if __name__ == "__main__" :

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--indir",        dest="indir",       help="Input folder with trained model",     default=None)
    parser.add_option("--tag",          dest="tag",         help="tag of the training folder",          default="")
    parser.add_option("--v",            dest="v",           help="Ntuple type ('ECAL' or 'HCAL')",      default='ECAL')
    parser.add_option("--eventLim",     dest="eventLim",    help="Maximum number of events to use",     default=None)
    parser.add_option("--addtag",       dest="addtag",      help="Add tag for different trainings",     default="")
    parser.add_option("--HoEcut",       dest="HoEcut",      help="Apply HoE cut at 0.95",               default=None)
    parser.add_option("--PlotRate",     dest="PlotRate",    help="Plots for rate proxy sample",         default=False,   action='store_true')
    (options, args) = parser.parse_args()
    print(options)

    indir = '/data_CMS/cms/motta/CaloL1calibraton/' + options.indir + '/' + options.v + 'training' + options.tag
    odir = '/data_CMS/cms/motta/CaloL1calibraton/' + options.indir + '/' + options.v + 'training' + options.tag + '/InputPlots' + options.addtag
    os.system('mkdir -p '+ odir)
    print('\n ### Reading TF records from: ' + indir + '/trainTFRecords/record_*.tfrecord')
    InTestRecords = glob.glob(indir+'/trainTFRecords/record_*.tfrecord')

    dataset = tf.data.TFRecordDataset(InTestRecords)
    batch_size = len(list(dataset))
    parsed_dataset = dataset.map(parse_function)
    data = parsed_dataset.batch(batch_size).as_numpy_iterator().next()
    print('\n ### N events: ' + str(len(list(dataset))))

    n_events = len(list(dataset))
    X = data[0]
    Y = data[1]

    iEta = np.argmax(X[:,:,2:], axis=2).reshape(-1)
    iEta = iEta[iEta != 0]
    L1Energy = np.sum(X[:,:,1], axis=1)
    JetEnergy = Y

    bins_res = np.linspace(0,3,200)
    plt.hist(L1Energy/JetEnergy, bins=bins_res, histtype='step', stacked=True, linewidth=2, color='black')
    plt.xlabel(r'Response')
    plt.ylabel('Entries')
    plt.grid(linestyle='dotted')
    plt.legend(fontsize=15, loc='upper left')
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
    savefile = odir + '/Response.png'
    plt.savefig(savefile)
    plt.close()

    bins_ieta = np.arange(iEta.min(), iEta.max()+1)
    plt.hist(iEta, bins=bins_ieta, histtype='step', stacked=True, linewidth=2, color='black')
    plt.xlabel(r'i$\eta$')
    plt.ylabel('Entries')
    plt.grid(linestyle='dotted')
    plt.legend(fontsize=15, loc='upper left')
    plt.yscale('log')
    mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
    savefile = odir + '/iEta.png'
    plt.savefig(savefile)
    plt.close()

    # plt.figure(figsize=(10,10))
    # sel_barrel = np.abs(df_jets['jetEta']) < 1.305
    # text_1 = leg_uncalib+r' barrel : $\mu={:.3f}, res={:.3f}$'.format(df_jets[sel_barrel]['unc_res'].mean(), df_jets[sel_barrel]['unc_res'].std()/df_jets[sel_barrel]['unc_res'].mean())
    # plt.hist(iEta, bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(0))
    # sel_endcap = np.abs(df_jets['jetEta']) >= 1.305
    # text_1 = leg_uncalib+r' endcap : $\mu={:.3f}, res={:.3f}$'.format(df_jets[sel_endcap]['unc_res'].mean(), df_jets[sel_endcap]['unc_res'].std()/df_jets[sel_endcap]['unc_res'].mean())
    # plt.hist(df_jets[sel_endcap]['unc_res'], bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(1))
    # text_1 = leg_uncalib+r': $\mu={:.3f}, res={:.3f}$'.format(df_jets['unc_res'].mean(), df_jets['unc_res'].std()/df_jets['unc_res'].mean())
    # counts, bins, _ = plt.hist(df_jets['unc_res'], bins=bins_res, label=text_1, histtype='step', stacked=True, linewidth=2, color=cmap(2))
    # plt.xlabel('Response')
    # plt.ylabel('a.u.')
    # plt.ylim(0, 1.3*np.max(counts))
    # plt.xlim(0,3)
    # plt.grid(linestyle='dotted')
    # plt.legend(fontsize=15, loc='upper left')
    # mplhep.cms.label(data=False, rlabel='(13.6 TeV)', fontsize=20)
    # # plt.title('Jets Resolution {}'.format(v_sample))
    # savefile = odir + '/Res_{}.png'.format(v_sample)
    # plt.savefig(savefile)
    # print(savefile)
    # plt.close()

    # if options.PlotRate:

    #     print('\n ### Reading TF records for Rate Plots from: ' + indir + '/rateTFRecords/record_*.tfrecord')
    #     InTestRecords = glob.glob(indir+'/rateTFRecords/record_*.tfrecord')[:options.filesLim]
    #     dataset = tf.data.TFRecordDataset(InTestRecords)
    #     batch_size = len(list(dataset))
    #     parsed_dataset = dataset.map(parse_function)
    #     data = parsed_dataset.batch(batch_size).as_numpy_iterator().next()
    #     print('\n ### N events: ' + str(len(list(dataset))))

    #     if options.eventLim:
    #         print('\n ### Reading {} events'.format(options.eventLim))
    #         n_events = int(options.eventLim)
    #         Towers = data[0][:int(options.eventLim)]
    #     else:
    #         n_events = len(list(dataset))
    #         Towers = data[0]

    #     print('\n ### Load Rate Dataframes')

    #     # Extract the iem and hcalET columns from Towers
    #     if options.v == 'ECAL':
    #         iem = Towers[:, :, 1].reshape(-1)
    #         hcalET = Towers[:, :, 0].reshape(-1)
    #     elif options.v == 'HCAL':
    #         iem = Towers[:, :, 0].reshape(-1)
    #         hcalET = Towers[:, :, 1].reshape(-1)
    #     # Extract the ieta column from Towers using argmax
    #     ieta = np.argmax(Towers[:, :, 2:], axis=2).reshape(-1) + 1
    #     # Create arrays for the id and jetPt columns
    #     id_arr = np.repeat(np.arange(len(Towers)), Towers.shape[1])
    #     iesum = (iem + hcalET)/2

    #     # Combine the arrays into a dictionary and create the dataframe
    #     df_Towers = pd.DataFrame({'id': id_arr, 'iem': iem, 'hcalET': hcalET, 'iesum': iesum, 'ieta': ieta})
    #     df_Towers.groupby('id').apply(lambda x: x.sort_values('iesum', ascending=False))

    #     # compute sum of the raw energy
    #     df_jets = pd.DataFrame()
    #     if options.v == 'HCAL':
    #         df_jets['jetPtRate']  = df_Towers.groupby('id').iesum.sum()
    #     if options.v == 'ECAL':
    #         df_jets['jetPtRate']  = df_Towers.groupby('id').iem.sum()
    #     df_jets['jetIEta']    = df_Towers.groupby('id').ieta.first()
    #     df_jets['jetSeed']    = df_Towers.groupby('id').iesum.max()

    #     PlotEtaIesum(df_Towers, odir, options.v, "Rate")

    #     PlotRate(df_jets, odir, options.v)
