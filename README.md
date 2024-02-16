# CaloL1CalibrationProducer

This repository contains all the packages and scripts to produce and test the Layer-1 Trigger Towers (TT) calibration.

### Introduction

This guide contains the instructions to extract the calibration from 2023 data to be applied to 2024 data taking.
It is divided into:
- [Installation](#introduction)
- [1. Re-emulate data with the latest data taking conditions](#1-re-emulate-data-with-the-latest-data-taking-conditions)
- [2. Read jets a prepare inputs](#2-read-jets-a-prepare-inputs)
- [3. Train the model and extract the Scale Factors](#3-train-the-model-and-extract-the-scale-factors)

## Installation

```bash
cmsrel CMSSW_13_3_0
cd CMSSW_13_3_0/src
cmsenv
git cms-init
git cms-addpkg L1Trigger/L1TCalorimeter
git cms-addpkg L1Trigger/L1TNtuples
git cms-addpkg L1Trigger/Configuration
git cms-addpkg L1Trigger/L1TGlobal
git cms-addpkg L1Trigger/L1TCommon
git cms-addpkg L1Trigger/L1TZDC
mkdir L1Trigger/L1TZDC/data
cd L1Trigger/L1TZDC/data
wget https://raw.githubusercontent.com/cms-data/L1Trigger-L1TCalorimeter/master/zdcLUT_HI_v0_1.txt
cd -
git clone https://github.com/cms-l1t-offline/L1Trigger-L1TCalorimeter.git L1Trigger/L1TCalorimeter/data
git clone git@github.com:elenavernazza/CaloL1CalibrationProducer.git
git cms-checkdeps -A -a
scram b -j 8 
cd CaloL1CalibrationProducer
```

## 1. Re-emulate data with the latest data taking conditions

The first step is to re-emulate data acquired during 2023 with the latest data taking conditions.
The data samples can be found on [CMSDAS](https://cmsweb.cern.ch/das/).

We will use either RAW or RAW-RECO for the re-emulation, the other versions do not contain enough information.
Since these formats are quite heavy, always check that the files are actually available and not on TAPE.

Three datasets will be considered:
- EGamma for the calibration of ECAL
- JetMET for the calibration of HCAL and HF
- ZeroBias for the rate simulation

Once the list of files for the three datasets is finalized, copy the list to a txt file inside the `L1NtupleLauncher/inputFiles` folder.

<details>
<summary>File list</summary>

- EGamma

```bash
dasgoclient --query=="file dataset=/EGamma0/Run2023B-ZElectron-PromptReco-v1/RAW-RECO" >> L1NtupleLauncher/inputFiles/EGamma__Run2023B-ZElectron-PromptReco-v1__RAW-RECO.txt
dasgoclient --query=="file dataset=/EGamma1/Run2023B-ZElectron-PromptReco-v1/RAW-RECO" >> L1NtupleLauncher/inputFiles/EGamma__Run2023B-ZElectron-PromptReco-v1__RAW-RECO.txt

dasgoclient --query=="file dataset=/EGamma0/Run2023C-ZElectron-PromptReco-v4/RAW-RECO" >> L1NtupleLauncher/inputFiles/EGamma__Run2023C-ZElectron-PromptReco-v4__RAW-RECO.txt
dasgoclient --query=="file dataset=/EGamma1/Run2023C-ZElectron-PromptReco-v4/RAW-RECO" >> L1NtupleLauncher/inputFiles/EGamma__Run2023C-ZElectron-PromptReco-v4__RAW-RECO.txt

dasgoclient --query=="file dataset=/EGamma0/Run2023D-ZElectron-PromptReco-v2/RAW-RECO" >> L1NtupleLauncher/inputFiles/EGamma__Run2023D-ZElectron-PromptReco-v2__RAW-RECO.txt
dasgoclient --query=="file dataset=/EGamma1/Run2023D-ZElectron-PromptReco-v2/RAW-RECO" >> L1NtupleLauncher/inputFiles/EGamma__Run2023D-ZElectron-PromptReco-v2__RAW-RECO.txt
```

- JetMET

```bash
dasgoclient --query=="file dataset=/JetMET0/Run2023B-PromptReco-v1/AOD" >> L1NtupleLauncher/inputFiles/JetMET__Run2023B-PromptReco-v1__AOD.txt
dasgoclient --query=="file dataset=/JetMET1/Run2023B-PromptReco-v1/AOD" >> L1NtupleLauncher/inputFiles/JetMET__Run2023B-PromptReco-v1__AOD.txt

dasgoclient --query=="file dataset=/JetMET0/Run2023C-PromptReco-v4/AOD" >> L1NtupleLauncher/inputFiles/JetMET__Run2023C-PromptReco-v4__AOD.txt
dasgoclient --query=="file dataset=/JetMET1/Run2023C-PromptReco-v4/AOD" >> L1NtupleLauncher/inputFiles/JetMET__Run2023C-PromptReco-v4__AOD.txt

dasgoclient --query=="file dataset=/JetMET0/Run2023D-PromptReco-v2/AOD" >> L1NtupleLauncher/inputFiles/JetMET__Run2023D-PromptReco-v2__AOD.txt
dasgoclient --query=="file dataset=/JetMET1/Run2023D-PromptReco-v2/AOD" >> L1NtupleLauncher/inputFiles/JetMET__Run2023D-PromptReco-v2__AOD.txt
```

- ZeroBias

```bash
dasgoclient --query=="file dataset=/EphemeralZeroBias0/Run2023D-v1/RAW" >> L1NtupleLauncher/inputFiles/EphemeralZeroBias__Run2023D-v1__RAW.txt
```

To check the availability of samples, use:

```bash
python3 FindAvalibaleFiles.py --sample /EGamma0/Run2023B-ZElectron-PromptReco-v1/RAW-RECO --txt EGamma__Run2023B-ZElectron-PromptReco-v1__RAW-RECO
python3 FindAvalibaleFiles.py --sample /EGamma1/Run2023B-ZElectron-PromptReco-v1/RAW-RECO --txt EGamma__Run2023B-ZElectron-PromptReco-v1__RAW-RECO
python3 FindAvalibaleFiles.py --sample /JetMET0/Run2023B-PromptReco-v1/AOD --txt JetMET__Run2023B-PromptReco-v1__AOD
python3 FindAvalibaleFiles.py --sample /JetMET1/Run2023B-PromptReco-v1/AOD --txt JetMET__Run2023B-PromptReco-v1__AOD
```

The EraD samples will be used for the performance evaluation (~30 files are enough):
```bash
python3 FindAvalibaleFiles.py --sample /EGamma0/Run2023B-ZElectron-PromptReco-v1/RAW-RECO --txt EGamma__Run2023D-ZElectron-PromptReco-v2__RAW-RECO_test # All but took 30
python3 FindAvalibaleFiles.py --sample /JetMET0/Run2023D-PromptReco-v2/AOD --txt JetMET__Run2023D-PromptReco-v2__AOD # 29
python3 FindAvalibaleFiles.py --sample /EphemeralZeroBias0/Run2023D-v1/RAW --txt EphemeralZeroBias__Run2023D-v1__RAW_test # All but took 30
```

</details>

The latest data taking conditions are defined by:
- CMSSW version (CMSSW_13_3_0)
- globalTag (130X_dataRun3_Prompt_v4)
- current caloParams file (caloParams_2023_v0_4_noL1Calib_cfi)
- certification json [reference](https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions23/PromptReco/Cert_Collisions2023_366442_370790_Golden.json)

Note: This re-emulation calcels all the old calibration applied, so check that all the SFs are 1 in the caloParams_2023_v0_4_noL1Calib_cfi (except for the Zero Suppression). If not, change them manually.

Copy your certification json to `/L1NtupleLauncher/DataCertificationJsons`.

Copy your caloParams_2023_v0_4_noL1Calib_cfi.py to `src/L1Trigger/L1TCalorimeter/python/`.

### Re-emulate EGamma

```bash
cd L1NtupleLauncher
voms-proxy-init --rfc --voms cms -valid 192:00
python submitOnTier3.py --inFileList EGamma__Run2023B-ZElectron-PromptReco-v1__RAW-RECO \
    --outTag GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json \
    --inJson Cert_Collisions2023_366442_370790_Golden \
    --caloParams caloParams_2023_v0_4_noL1Calib_cfi \
    --globalTag 130X_dataRun3_Prompt_v4 \
    --nJobs 1344 --queue short --maxEvts -1 --data --recoFromSKIM
python submitOnTier3.py --inFileList EGamma__Run2023C-ZElectron-PromptReco-v4__RAW-RECO \
    --outTag GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json \
    --inJson Cert_Collisions2023_366442_370790_Golden \
    --caloParams caloParams_2023_v0_4_noL1Calib_cfi \
    --globalTag 130X_dataRun3_Prompt_v4 \
    --nJobs 4000 --queue short --maxEvts -1 --data --recoFromSKIM
```

### Re-emulate JetMET

```bash
cd L1NtupleLauncher
voms-proxy-init --rfc --voms cms -valid 192:00
python submitOnTier3.py --inFileList JetMET__Run2023B-PromptReco-v1__AOD \
    --outTag GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json \
    --inJson Cert_Collisions2023_366442_370790_Golden \
    --caloParams caloParams_2023_v0_4_noL1Calib_cfi \
    --globalTag 130X_dataRun3_Prompt_v4 \
    --nJobs 5828 --queue short --maxEvts -1 --data --recoFromAOD
```

<!-- ### Re-emulate data ZeroBias

```bash
cd L1NtupleLauncher
voms-proxy-init --rfc --voms cms -valid 192:00

python submitOnTier3.py --inFileList EphemeralZeroBias__Run2023D-v1__RAW \
    --outTag GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data \
    --inJson Cert_Collisions2023_366442_370790_Golden \
    --caloParams caloParams_2023_v0_4_noL1Calib_cfi \
    --globalTag 130X_dataRun3_Prompt_v4 \
    --nJobs 772 --queue short --maxEvts -1 --data
``` -->

<details>
<summary>Check samples</summary>

Since many files are on TAPE, some jobs will fail due to error opening the file.
To only select the good files and eventually resubmit non-finished jobs use:

```bash
python3 resubmit_Unfinished.py /data_CMS/cms/motta/CaloL1calibraton/L1NTuples/EGamma__Run2023B-ZElectron-PromptReco-v1__RAW-RECO__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json
python3 resubmit_Unfinished.py /data_CMS/cms/motta/CaloL1calibraton/L1NTuples/EGamma__Run2023C-ZElectron-PromptReco-v4__RAW-RECO__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json
```
```bash
python3 resubmit_Unfinished.py /data_CMS/cms/motta/CaloL1calibraton/L1NTuples/JetMET__Run2023B-PromptReco-v1__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json
```

You can plot the re-emulated samples using:

```bash
cd L1Plotting
python3 resolutions.py --indir EGamma__Run2023B-ZElectron-PromptReco-v1__RAW-RECO__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json/GoodNtuples \
 --outdir 2024_02_15_NtuplesV58/TestInput_EGamma2023B --label EGamma_data_reco --reco --nEvts 50000 --target ele \
 --raw --LooseEle --do_EoTot --tag _LooseEle_50K_Raw
```
```bash
python3 resolutions.py --indir JetMET__Run2023B-PromptReco-v1__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json/GoodNtuples \
 --outdir 2024_02_15_NtuplesV58/TestInput_JetMET2023B --label Jet_data_reco --reco --nEvts 50000 --target jet \
 --raw --PuppiJet --jetPtcut 30 --do_HoTot --tag _PuppiJet_50K_Pt30_Raw
```

</details>

## 2. Read jets a prepare inputs

At this point we can read the re-emulated samples to extract the 9x9 chunky donut describing the EGamma and Jets at Layer-1.

The reader will loop over all the jets and save the input to `*.npz` tensors directly compatible with the training script.
Some selections are applied:
- DeltaR separation between two objects
- Matching between L1 object and Offline object
- Electromagnetic (ecalcut) or Hadronic (hcalcut) fraction
- For EGamma, only electrons passing the LooseEle flag are considered
- For Jets, minimum JetPt at 30 GeV
- If specified, maximum JetPt and maximum Eta 

### Read EGamma

```bash
cd L1NtupleReader
python3 batchSubmitOnTier3.py --indir /data_CMS/cms/motta/CaloL1calibraton/L1NTuples/EGamma__Run2023B-ZElectron-PromptReco-v1__RAW-RECO__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json/GoodNtuples \
    --outdir /data_CMS/cms/motta/CaloL1calibraton/2024_02_15_NtuplesV58/EGamma_Run2023B_LooseEle_EoTot80 \
    --target reco --type ele --chunk_size 5000 \
    --queue short \
    --ecalcut 0.80 --applyCut_3_6_9 True --LooseEle --matching
python3 batchSubmitOnTier3.py --indir /data_CMS/cms/motta/CaloL1calibraton/L1NTuples/EGamma__Run2023C-ZElectron-PromptReco-v4__RAW-RECO__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json/GoodNtuples \
    --outdir /data_CMS/cms/motta/CaloL1calibraton/2024_02_15_NtuplesV58/EGamma_Run2023C_LooseEle_EoTot80 \
    --target reco --type ele --chunk_size 5000 \
    --queue short \
    --ecalcut 0.80 --applyCut_3_6_9 True --LooseEle --matching
```

### Read Jet

```bash
cd L1NtupleReader
python3 batchSubmitOnTier3.py --indir /data_CMS/cms/motta/CaloL1calibraton/L1NTuples/JetMET__Run2023B-PromptReco-v1__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json/GoodNtuples \
    --outdir /data_CMS/cms/motta/CaloL1calibraton/2024_02_15_NtuplesV58/JetMET_Run2023B_PuppiJet_Pt30_HoTot70 \
    --target reco --type jet --chunk_size 5000 \
    --queue short \
    --hcalcut 0.70 --lJetPtCut 30 --PuppiJet --matching
python3 batchSubmitOnTier3.py --indir /data_CMS/cms/motta/CaloL1calibraton/L1NTuples/JetMET__Run2023B-PromptReco-v1__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json/GoodNtuples \
    --outdir /data_CMS/cms/motta/CaloL1calibraton/2024_02_15_NtuplesV58/JetMET_Run2023B_PuppiJet_BarrelEndcap_Pt30_HoTot70 \
    --target reco --type jet --chunk_size 5000 \
    --queue short \
    --hcalcut 0.70 --lJetPtCut 30 --PuppiJet --matching --etacut 28
```

## 3. Train the model and extract the Scale Factors

Before training the model, check the distribution of the input towers with:

```bash
cd L1JaxTraining
python3 TowersJetsCounter.py --indir 2024_02_15_NtuplesV58/EGamma_Run2023*_LooseEle_EoTot80/GoodNtuples/tensors \
    --odir Trainings_2023/TestInputEGamma --jetsLim 1000000
python3 TowersJetsCounter.py --indir 2024_02_15_NtuplesV58/JetMET_Run2023B_PuppiJet_Pt30_HoTot70/GoodNtuples/tensors \
    --odir Trainings_2023/TestInputJetMET --jetsLim 1000000
python3 TowersJetsCounter.py --indir 2024_02_15_NtuplesV58/JetMET_Run2023B_PuppiJet_BarrelEndcap_Pt30_HoTot70/GoodNtuples/tensors \
    --odir Trainings_2023/TestInputJetMET_BarrelEndcap --jetsLim 1000000
```

### Calibrate ECAL

### Calibrate HCAL

```bash
cd L1JaxTraining
python3 JaxOptimizer.py --indir 2024_02_15_NtuplesV58/JetMET_Run2023B_PuppiJet_Pt30_HoTot70/GoodNtuples/tensors \
    --odir Trainings_2023/JAX_HCAL_1 --jetsLim 1000000 --lr 0.5 --bs 4096 --ep 100 --scale 0.75 --v HCAL
```

The full list of training is stored in `L1JaxTraining/Instructions/RunTrainings.sh`.

## 4. Re-emulate and compare with old calibration

```bash
cd L1JaxTraining
voms-proxy-init --rfc --voms cms -valid 192:00
source Instructions/TestsTraining.sh JAX_HCAL_1
```

<details>
<summary>Full commands</summary>

```bash
cd L1JaxTraining
voms-proxy-init --rfc --voms cms -valid 192:00
python3 SFPlots.py --indir Trainings_2023/JAX_HCAL_1
python3 ProduceCaloParams.py --name caloParams_2023_JAX_HCAL_1_newCalib_cfi --base caloParams_2023_v0_4_noL1Calib_cfi.py \
    --HCAL ./Trainings_2023/JAX_HCAL_1/ScaleFactors_HCAL.csv --HF ./Trainings_2023/JAX_HCAL_1/ScaleFactors_HCAL.csv

python3 RDF_ResolutionFast.py --indir JetMET__Run2023D-PromptReco-v2__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json/GoodNtuples \
 --reco --target jet --do_HoTot --raw --PuppiJet --jetPtcut 30 --nEvts 100000 --no_plot \
 --HCALcalib --caloParam caloParams_2023_v0_4_cfi.py \
 --outdir Trainings_2023/JAX_HCAL_0/NtuplesVold
python3 RDF_ResolutionFast.py --indir JetMET__Run2023D-PromptReco-v2__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json/GoodNtuples \
 --reco --target jet --do_HoTot --raw --PuppiJet --jetPtcut 30 --nEvts 100000 --no_plot \
 --HCALcalib --caloParam caloParams_2023_v0_4_noL1Calib_cfi.py \
 --outdir Trainings_2023/JAX_HCAL_0/NtuplesVunc
python3 RDF_ResolutionFast.py --indir JetMET__Run2023D-PromptReco-v2__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json/GoodNtuples \
 --reco --target jet --do_HoTot --raw --PuppiJet --jetPtcut 30 --nEvts 100000 --no_plot \
 --HCALcalib --caloParam caloParams_2023_JAX_HCAL_1_newCalib_cfi.py \
 --outdir Trainings_2023/JAX_HCAL_1/NtuplesVnew

python3 comparisonPlotsFast.py --indir Trainings_2023/JAX_HCAL_1/NtuplesVnew --target jet --reco \
 --old Trainings_2023/JAX_HCAL_0/NtuplesVold --unc Trainings_2023/JAX_HCAL_0/NtuplesVunc \
 --do_HoTot --doRate False --doTurnOn False
```

#### Re-emulate JetMET

```bash
python3 submitOnTier3.py --inFileList EphemeralZeroBias__Run2023D-v1__RAW \
    --outTag GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data \
    --nJobs 30 --queue short --maxEvts 5000 \
    --globalTag 130X_dataRun3_Prompt_v4 --data \
    --caloParams caloParams_2023_v0_4_noL1Calib_cfi
python3 submitOnTier3.py --inFileList EphemeralZeroBias__Run2023D-v1__RAW \
    --outTag GT130XdataRun3Promptv4_CaloParams2023v04_data \
    --nJobs 30 --queue short --maxEvts 5000 \
    --globalTag 130X_dataRun3_Prompt_v4 --data \
    --caloParams caloParams_2023_v0_4_cfi
python3 submitOnTier3.py --inFileList EphemeralZeroBias__Run2023D-v1__RAW \
 --outTag GT130XdataRun3Promptv4_CaloParams2023JAX_HCAL_1_data \
 --nJobs 30 --queue short --maxEvts 5000 \
 --globalTag 130X_dataRun3_Prompt_v4 --data \
 --caloParams caloParams_2023_JAX_HCAL_1_newCalib_cfi

python3 submitOnTier3.py --inFileList JetMET__Run2023D-PromptReco-v2__AOD \
    --outTag GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json \
    --inJson Cert_Collisions2023_366442_370790_Golden \
    --nJobs 29 --queue short --maxEvts 5000 \
    --globalTag 130X_dataRun3_Prompt_v4 --data --recoFromAOD \
    --caloParams caloParams_2023_v0_4_noL1Calib_cfi
python3 submitOnTier3.py --inFileList JetMET__Run2023D-PromptReco-v2__AOD \
    --outTag GT130XdataRun3Promptv4_CaloParams2023v04_data_reco_json \
    --inJson Cert_Collisions2023_366442_370790_Golden \
    --nJobs 29 --queue short --maxEvts 5000 \
    --globalTag 130X_dataRun3_Prompt_v4 --data --recoFromAOD \
    --caloParams caloParams_2023_v0_4_cfi
python3 submitOnTier3.py --inFileList JetMET__Run2023D-PromptReco-v2__AOD \
 --outTag GT130XdataRun3Promptv4_CaloParams2023JAX_HCAL_1_data_reco_json \
 --inJson Cert_Collisions2023_366442_370790_Golden \
 --nJobs 29 --queue short --maxEvts 5000 \
 --globalTag 130X_dataRun3_Prompt_v4 --data --recoFromAOD \
 --caloParams caloParams_2023_JAX_HCAL_1_newCalib_cfi
```

```bash
python3 resubmit_Unfinished.py /data_CMS/cms/motta/CaloL1calibraton/L1NTuples/JetMET__Run2023D-PromptReco-v2__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json
python3 resubmit_Unfinished.py /data_CMS/cms/motta/CaloL1calibraton/L1NTuples/JetMET__Run2023D-PromptReco-v2__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_data_reco_json
```

#### Re-emulate EGamma

```bash
cd L1NtupleLauncher
voms-proxy-init --rfc --voms cms -valid 192:00
python submitOnTier3.py --inFileList EGamma__Run2023D-ZElectron-PromptReco-v2__RAW-RECO \
    --outTag GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json \
    --inJson Cert_Collisions2023_366442_370790_Golden \
    --globalTag 130X_dataRun3_Prompt_v4 \
    --nJobs 30 --queue short --maxEvts -1 --data --recoFromSKIM \
    --caloParams caloParams_2023_v0_4_noL1Calib_cfi
python submitOnTier3.py --inFileList EGamma__Run2023D-ZElectron-PromptReco-v2__RAW-RECO \
    --outTag GT130XdataRun3Promptv4_CaloParams2023v04_data_reco_json \
    --inJson Cert_Collisions2023_366442_370790_Golden \
    --globalTag 130X_dataRun3_Prompt_v4 \
    --nJobs 30 --queue short --maxEvts -1 --data --recoFromSKIM \
    --caloParams caloParams_2023_v0_4_cfi
```

```bash
python3 resubmit_Unfinished.py /data_CMS/cms/motta/CaloL1calibraton/L1NTuples/EGamma__Run2023D-ZElectron-PromptReco-v2__RAW-RECO__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json
```

</details>

## Plot HCAL calibration

```bash
cd L1JaxTraining
source Instructions/TestsPerformance.sh JAX_HCAL_1
```

<details>
<summary>Full commands</summary>

```bash
cd L1Plotting
python3 rate.py --indir EphemeralZeroBias__Run2023D-v1__RAW__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data \
 --outdir 2024_02_15_NtuplesV58/JAX_HCAL/NtuplesVuncL1pt --target jet --raw --nEvts 100000 --no_plot

python3 rate.py --indir EphemeralZeroBias__Run2023D-v1__RAW__GT130XdataRun3Promptv4_CaloParams2023v04_data \
 --outdir 2024_02_15_NtuplesV58/JAX_HCAL/NtuplesVoldL1pt --target jet --raw --nEvts 100000 --no_plot

python3 turnOn.py --indir JetMET__Run2023D-PromptReco-v2__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json/GoodNtuples \
 --outdir 2024_02_15_NtuplesV58/JAX_HCAL/NtuplesVuncL1pt --reco --target jet --raw --PuppiJet --nEvts 100000

python3 turnOn.py --indir JetMET__Run2023D-PromptReco-v2__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_data_reco_json/GoodNtuples \
 --outdir 2024_02_15_NtuplesV58/JAX_HCAL/NtuplesVoldL1pt --reco --target jet --raw --PuppiJet --nEvts 100000

python3 resolutions.py --indir JetMET__Run2023D-PromptReco-v2__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data_reco_json/GoodNtuples \
 --outdir 2024_02_15_NtuplesV58/JAX_HCAL/NtuplesVuncL1pt --reco --target jet --raw --PuppiJet --jetPtcut 30 --nEvts 100000 --no_plot

python3 resolutions.py --indir JetMET__Run2023D-PromptReco-v2__AOD__GT130XdataRun3Promptv4_CaloParams2023v04_data_reco_json/GoodNtuples \
 --outdir 2024_02_15_NtuplesV58/JAX_HCAL/NtuplesVoldL1pt --reco --target jet --raw --PuppiJet --jetPtcut 30 --nEvts 100000 --no_plot

python3 rate.py --indir EphemeralZeroBias__Run2023D-v1__RAW__GT130XdataRun3Promptv4_CaloParams2023v04_noL1Calib_data \
 --outdir 2024_02_15_NtuplesV58/JAX_HCAL/NtuplesVuncL1pt --target jet --raw --nEvts 100000 --no_plot --offline

python3 rate.py --indir EphemeralZeroBias__Run2023D-v1__RAW__GT130XdataRun3Promptv4_CaloParams2023v04_data \
 --outdir 2024_02_15_NtuplesV58/JAX_HCAL/NtuplesVoldL1pt --target jet --raw --nEvts 100000 --no_plot --offline
```

</details>