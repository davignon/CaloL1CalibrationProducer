import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloParams_cfi import caloParamsSource
import L1Trigger.L1TCalorimeter.caloParams_cfi
caloStage2Params = L1Trigger.L1TCalorimeter.caloParams_cfi.caloParams.clone(

    # EG
    egHcalThreshold            = 0.,
    egTrimmingLUTFile          = "L1Trigger/L1TCalorimeter/data/egTrimmingLUT_10_v16.01.19.txt",
    egHOverEcutBarrel          = 3,
    egHOverEcutEndcap          = 4,
    egBypassExtHOverE          = 0,
    egMaxHOverELUTFile         = "L1Trigger/L1TCalorimeter/data/HoverEIdentification_0.995_v15.12.23.txt",
    egCompressShapesLUTFile    = "L1Trigger/L1TCalorimeter/data/egCompressLUT_v4.txt",
    egShapeIdType              = "compressed",
    egShapeIdLUTFile           = "L1Trigger/L1TCalorimeter/data/shapeIdentification_adapt0.99_compressedieta_compressedE_compressedshape_v15.12.08.txt", #Not used any more in the current emulator version, merged with calibration LUT

    egIsolationType            = "compressed",
    egIsoLUTFile               = "L1Trigger/L1TCalorimeter/data/EG_Iso_LUT_Flat_WP_v2_Tight1358_20p0_0p7_40p0_v1_APR23.txt",
    egIsoLUTFile2              = "L1Trigger/L1TCalorimeter/data/EG_Iso_LUT_Flat_WP_v2_Loose610_10p0_0p7_40p0_v1_APR23.txt",

    egIsoVetoNrTowersPhi       = 2,
    egPUSParams                = cms.vdouble(1,4,32), #Isolation window in firmware goes up to abs(ieta)=32 for now
    egCalibrationType          = "compressed",
    egCalibrationVersion       = 0,
    egCalibrationLUTFile       = "L1Trigger/L1TCalorimeter/data/EG_Calibration_LUT_correctedEtCalibLUT_v1_APR2023.txt",

    # Tau
    isoTauEtaMax               = 25,
    tauSeedThreshold           = 0.,
    tauIsoLUTFile              = "L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_2023_calibThr1p7_V2gs_effMin0p9_eMin16_eMax60.txt",
    tauIsoLUTFile2             = "L1Trigger/L1TCalorimeter/data/Tau_Iso_LUT_2023_calibThr1p7_V2gs_effMin0p9_eMin16_eMax60.txt",
    tauCalibrationLUTFile      = "L1Trigger/L1TCalorimeter/data/Tau_Cal_LUT_2023_calibThr1p7_V2.txt",
    tauCompressLUTFile         = "L1Trigger/L1TCalorimeter/data/tauCompressAllLUT_12bit_v3.txt",
    tauPUSParams               = [1,4,32],

    # jets
    jetSeedThreshold           = 4.0,
    jetPUSType                 = "ChunkyDonut",

    # Calibration options
    jetCalibrationType         = "LUT",
    jetCompressPtLUTFile       = "L1Trigger/L1TCalorimeter/data/lut_pt_compress_2017v1.txt",
    jetCompressEtaLUTFile      = "L1Trigger/L1TCalorimeter/data/lut_eta_compress_2017v1.txt",
    jetCalibrationLUTFile      = "L1Trigger/L1TCalorimeter/data/lut_calib_2022v5_ECALZS_noHFJEC.txt",


    # sums: 0=ET, 1=HT, 2=MET, 3=MHT
    etSumEtaMin             = [1, 1, 1, 1, 1],
    etSumEtaMax             = [28,  26, 28,  26, 28],
    etSumEtThreshold        = [0.,  30.,  0.,  30., 0.], # only 2nd (HT) and 4th (MHT) values applied
    etSumMetPUSType         = "LUT", # et threshold from this LUT supercedes et threshold in line above
    etSumBypassEttPUS       = 1,
    etSumBypassEcalSumPUS   = 1,

    etSumMetPUSLUTFile               = "L1Trigger/L1TCalorimeter/data/metPumLUT_2022_HCALOff_p5.txt",


    # Layer 1 SF
    layer1ECalScaleETBins = cms.vint32([3, 6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256]),
    layer1ECalScaleFactors = cms.vdouble([
        1.12, 1.13, 1.13, 1.12, 1.12, 1.12, 1.13, 1.12, 1.13, 1.12, 1.13, 1.13, 1.14, 1.13, 1.13, 1.13, 1.14, 1.26, 1.11, 1.20, 1.21, 1.22, 1.19, 1.20, 1.19, 0.00, 0.00, 0.00,
        1.12, 1.13, 1.13, 1.12, 1.12, 1.12, 1.13, 1.12, 1.13, 1.12, 1.13, 1.13, 1.14, 1.13, 1.13, 1.13, 1.14, 1.26, 1.11, 1.20, 1.21, 1.22, 1.19, 1.20, 1.19, 1.22, 0.00, 0.00,
        1.08, 1.09, 1.08, 1.08, 1.11, 1.08, 1.09, 1.09, 1.09, 1.09, 1.15, 1.09, 1.10, 1.10, 1.10, 1.10, 1.10, 1.23, 1.07, 1.15, 1.14, 1.16, 1.14, 1.14, 1.15, 1.14, 1.14, 0.00, 
        1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.06, 1.07, 1.07, 1.07, 1.07, 1.07, 1.08, 1.07, 1.09, 1.08, 1.17, 1.06, 1.11, 1.10, 1.13, 1.10, 1.10, 1.11, 1.11, 1.11, 1.09, 
        1.04, 1.05, 1.04, 1.05, 1.04, 1.05, 1.06, 1.06, 1.05, 1.05, 1.05, 1.06, 1.06, 1.06, 1.06, 1.06, 1.07, 1.15, 1.04, 1.09, 1.09, 1.10, 1.09, 1.09, 1.10, 1.10, 1.10, 1.08, 
        1.04, 1.03, 1.04, 1.04, 1.04, 1.04, 1.04, 1.04, 1.04, 1.04, 1.04, 1.04, 1.05, 1.06, 1.04, 1.05, 1.05, 1.13, 1.03, 1.07, 1.08, 1.08, 1.08, 1.07, 1.07, 1.09, 1.08, 1.07, 
        1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.04, 1.04, 1.05, 1.05, 1.05, 1.05, 1.05, 1.12, 1.03, 1.06, 1.06, 1.08, 1.07, 1.07, 1.06, 1.08, 1.07, 1.06, 
        1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.04, 1.04, 1.04, 1.04, 1.04, 1.03, 1.10, 1.02, 1.05, 1.06, 1.06, 1.06, 1.06, 1.05, 1.06, 1.06, 1.06, 
        1.02, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.04, 1.03, 1.03, 1.02, 1.07, 1.02, 1.04, 1.04, 1.05, 1.06, 1.05, 1.05, 1.06, 1.06, 1.05, 
        1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.09, 1.02, 1.04, 1.05, 1.05, 1.05, 1.05, 1.04, 1.05, 1.06, 1.05, 
        1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.08, 1.01, 1.04, 1.04, 1.05, 1.05, 1.04, 1.04, 1.05, 1.06, 1.05, 
        1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.02, 1.01, 1.02, 1.02, 1.02, 1.02, 1.03, 1.03, 1.03, 1.03, 1.03, 1.06, 1.01, 1.04, 1.04, 1.05, 1.04, 1.03, 1.03, 1.04, 1.05, 1.04, 
        1.01, 1.00, 1.01, 1.01, 1.01, 1.01, 1.01, 1.00, 1.01, 1.02, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.03, 1.04, 1.01, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.00, 1.01, 
        1.02, 1.00, 1.00, 1.02, 1.00, 1.01, 1.01, 1.00, 1.00, 1.02, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.02, 1.04, 1.01, 1.03, 1.03, 1.03, 1.03, 1.02, 1.02, 1.02, 1.00, 1.01
    ]),

    layer1HCalScaleETBins = cms.vint32([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 256]),
    layer1HCalScaleFactors = cms.vdouble([
        0.0000,0.0000,0.0000,1.0000,0.0000,0.0000,1.0000,0.0000,1.0000,1.0000,2.0000,2.0000,3.0000,4.0000,4.0000,3.0000,2.0000,2.0000,2.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,
        1.0000,0.6667,1.0000,1.0000,0.6667,0.6667,1.0000,1.0000,1.0000,1.3334,1.3334,1.3334,2.0000,2.0000,2.0000,2.0000,1.6667,1.3334,1.3334,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,
        1.0000,1.0000,1.0000,1.0000,0.8000,0.8000,1.0000,1.0000,1.2000,1.2000,1.4000,1.4000,1.6000,1.8000,1.6000,1.6000,1.4000,1.2000,1.2000,1.0000,1.2000,1.0000,1.2000,1.0000,1.2000,1.2000,1.0000,1.0000,
        1.0000,1.0000,1.1429,1.1429,1.0000,1.0000,1.1429,1.0000,1.1429,1.2858,1.2858,1.2858,1.5715,1.5715,1.5715,1.4286,1.2858,1.2858,1.2858,1.1429,1.1429,1.1429,1.1429,1.1429,1.1429,1.1429,1.1429,1.1429,
        1.1112,1.0000,1.1112,1.1112,1.0000,1.0000,1.1112,1.1112,1.2223,1.2223,1.3334,1.3334,1.4445,1.4445,1.4445,1.4445,1.3334,1.2223,1.2223,1.1112,1.1112,1.1112,1.1112,1.1112,1.2223,1.1112,1.1112,1.1112,
        1.0910,1.0910,1.0910,1.0910,1.0000,1.0000,1.1819,1.0910,1.1819,1.1819,1.2728,1.2728,1.3637,1.4546,1.4546,1.3637,1.2728,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.0910,1.1819,1.1819,1.1819,1.1819,
        1.1539,1.0770,1.1539,1.1539,1.0770,1.0770,1.1539,1.1539,1.1539,1.2308,1.2308,1.2308,1.3847,1.3847,1.3847,1.3847,1.3077,1.2308,1.2308,1.1539,1.1539,1.1539,1.1539,1.1539,1.1539,1.1539,1.1539,1.1539,
        1.1334,1.1334,1.1334,1.1334,1.0667,1.0667,1.1334,1.1334,1.2000,1.2000,1.2667,1.2667,1.3334,1.4000,1.3334,1.3334,1.2667,1.2000,1.2000,1.1334,1.2000,1.1334,1.2000,1.1334,1.2000,1.2000,1.1334,1.1334,
        1.1177,1.1177,1.1765,1.1765,1.1177,1.1177,1.1765,1.1177,1.1765,1.2353,1.2353,1.2353,1.2942,1.3530,1.3530,1.2942,1.2353,1.2353,1.2353,1.1765,1.1765,1.1765,1.1765,1.1765,1.1765,1.1765,1.1765,1.1765,
        1.1579,1.1053,1.1579,1.1579,1.1053,1.1053,1.1579,1.1579,1.2106,1.2106,1.2632,1.2632,1.3158,1.3158,1.3158,1.3158,1.2632,1.2106,1.2106,1.1579,1.1579,1.1579,1.1579,1.1579,1.2106,1.1579,1.1579,1.1579,
        1.1429,1.1429,1.1429,1.1429,1.0953,1.0953,1.1905,1.1429,1.1905,1.1905,1.2381,1.2381,1.2858,1.3334,1.3334,1.2858,1.2381,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1429,1.1905,1.1905,1.1905,1.1905,
        1.1740,1.1305,1.1740,1.1740,1.1305,1.1305,1.1740,1.1740,1.1740,1.2174,1.2174,1.2174,1.3044,1.3044,1.3044,1.3044,1.2609,1.2174,1.2174,1.1740,1.1740,1.1740,1.1740,1.1740,1.1740,1.1740,1.1740,1.1740,
        1.1601,1.1601,1.1601,1.1601,1.1200,1.1200,1.1601,1.1601,1.2000,1.2000,1.2400,1.2400,1.2800,1.3200,1.2800,1.2800,1.2400,1.2000,1.2000,1.1601,1.2000,1.1601,1.2000,1.1601,1.2000,1.2000,1.1601,1.1601,
        1.1482,1.1482,1.1852,1.1482,1.1482,1.1482,1.1852,1.1482,1.1852,1.1852,1.2223,1.2223,1.2593,1.2963,1.2963,1.2593,1.2223,1.2223,1.2223,1.1852,1.1852,1.1852,1.1852,1.1852,1.1852,1.1852,1.1852,1.1852,
        1.1725,1.1380,1.1725,1.1725,1.1380,1.1380,1.1725,1.1725,1.2069,1.2069,1.2414,1.2414,1.2759,1.2759,1.2759,1.2759,1.2414,1.2069,1.2069,1.1725,1.1725,1.1725,1.1725,1.1725,1.1725,1.1725,1.1725,1.1725,
        1.1613,1.1613,1.1613,1.1613,1.1291,1.1291,1.1936,1.1613,1.1936,1.1936,1.2259,1.2259,1.2581,1.2904,1.2904,1.2581,1.2259,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1613,1.1936,1.1936,1.1936,1.1936,
        1.1819,1.1516,1.1819,1.1819,1.1516,1.1516,1.1819,1.1819,1.1819,1.2122,1.2122,1.2122,1.2728,1.2728,1.2728,1.2728,1.2425,1.2122,1.2122,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,
        1.1715,1.1715,1.1715,1.1715,1.1429,1.1429,1.1715,1.1715,1.2000,1.2000,1.2286,1.2286,1.2572,1.2858,1.2572,1.2572,1.2286,1.2000,1.2000,1.1715,1.2000,1.1715,1.2000,1.1715,1.2000,1.2000,1.1715,1.1715,
        1.1622,1.1622,1.1892,1.1622,1.1622,1.1622,1.1892,1.1622,1.1892,1.1892,1.2163,1.2163,1.2433,1.2703,1.2703,1.2433,1.2163,1.2163,1.2163,1.1892,1.1892,1.1892,1.1892,1.1892,1.1892,1.1892,1.1892,1.1892,
        1.1795,1.1539,1.1795,1.1795,1.1539,1.1539,1.1795,1.1795,1.2052,1.2052,1.2308,1.2308,1.2565,1.2565,1.2565,1.2565,1.2308,1.2052,1.2052,1.1795,1.1795,1.1795,1.1795,1.1795,1.1795,1.1795,1.1795,1.1795,
        1.1708,1.1708,1.1708,1.1708,1.1464,1.1464,1.1952,1.1708,1.1952,1.1952,1.2196,1.2196,1.2440,1.2683,1.2683,1.2440,1.2196,1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,1.1708,1.1952,1.1952,1.1952,1.1952,
        1.1628,1.1628,1.1861,1.1861,1.1628,1.1628,1.1861,1.1861,1.1861,1.2094,1.2094,1.2094,1.2559,1.2559,1.2559,1.2559,1.2326,1.2094,1.2094,1.1861,1.1861,1.1861,1.1861,1.1861,1.1861,1.1861,1.1861,1.1861,
        1.1778,1.1778,1.1778,1.1778,1.1556,1.1556,1.1778,1.1778,1.2000,1.2000,1.2223,1.2223,1.2445,1.2667,1.2445,1.2445,1.2223,1.2000,1.2000,1.1778,1.2000,1.1778,1.1778,1.1778,1.2000,1.2000,1.1778,1.1778,
        1.1703,1.1703,1.1915,1.1703,1.1490,1.1703,1.1915,1.1703,1.1915,1.1915,1.2128,1.2128,1.2341,1.2554,1.2554,1.2341,1.2128,1.2128,1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,
        1.1837,1.1633,1.1837,1.1837,1.1633,1.1633,1.1837,1.1837,1.2041,1.2041,1.2041,1.2245,1.2449,1.2449,1.2449,1.2449,1.2245,1.2041,1.2041,1.1837,1.1837,1.1837,1.1837,1.1837,1.1837,1.1837,1.1837,1.1837,
        1.1765,1.1765,1.1765,1.1765,1.1569,1.1569,1.1961,1.1765,1.1961,1.1961,1.2157,1.2157,1.2353,1.2550,1.2550,1.2353,1.2157,1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,1.1765,1.1961,1.1961,1.1961,1.1961,
        1.1699,1.1699,1.1887,1.1887,1.1699,1.1699,1.1887,1.1699,1.1887,1.2076,1.2076,1.2076,1.2453,1.2453,1.2453,1.2453,1.2076,1.2076,1.2076,1.1887,1.1887,1.1887,1.1887,1.1887,1.1887,1.1887,1.1887,1.1887,
        1.1819,1.1819,1.1819,1.1819,1.1637,1.1637,1.1819,1.1819,1.2000,1.2000,1.2182,1.2182,1.2364,1.2364,1.2364,1.2364,1.2182,1.2000,1.2000,1.1819,1.2000,1.1819,1.1819,1.1819,1.2000,1.2000,1.1819,1.1819,
        1.1755,1.1755,1.1755,1.1755,1.1579,1.1579,1.1930,1.1755,1.1930,1.1930,1.2106,1.2106,1.2281,1.2457,1.2457,1.2281,1.2106,1.2106,1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,
        1.1865,1.1695,1.1865,1.1865,1.1695,1.1695,1.1865,1.1865,1.2034,1.2034,1.2034,1.2204,1.2373,1.2373,1.2373,1.2373,1.2204,1.2034,1.2034,1.1865,1.1865,1.1865,1.1865,1.1865,1.1865,1.1865,1.1865,1.1865,
        1.1804,1.1804,1.1804,1.1804,1.1640,1.1640,1.1804,1.1804,1.1968,1.1968,1.2132,1.2132,1.2296,1.2460,1.2460,1.2296,1.2132,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,1.1804,1.1968,1.1968,1.1968,1.1968,
        1.1747,1.1747,1.1905,1.1905,1.1747,1.1747,1.1905,1.1747,1.1905,1.2064,1.2064,1.2064,1.2223,1.2381,1.2381,1.2223,1.2064,1.2064,1.2064,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,
        1.1847,1.1847,1.1847,1.1847,1.1693,1.1693,1.1847,1.1847,1.2000,1.2000,1.2154,1.2154,1.2308,1.2308,1.2308,1.2308,1.2154,1.2000,1.2000,1.1847,1.2000,1.1847,1.1847,1.1847,1.2000,1.2000,1.1847,1.1847,
        1.1792,1.1792,1.1792,1.1792,1.1642,1.1642,1.1941,1.1792,1.1941,1.1941,1.2090,1.2090,1.2239,1.2389,1.2389,1.2239,1.2090,1.2090,1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,
        1.1885,1.1740,1.1885,1.1885,1.1740,1.1740,1.1885,1.1885,1.2029,1.2029,1.2029,1.2174,1.2319,1.2319,1.2319,1.2319,1.2174,1.2029,1.2029,1.1885,1.1885,1.1885,1.1885,1.1885,1.1885,1.1885,1.1885,1.1885,
        1.1831,1.1831,1.1831,1.1831,1.1691,1.1691,1.1831,1.1831,1.1972,1.1972,1.2113,1.2113,1.2254,1.2395,1.2395,1.2254,1.2113,1.1972,1.1972,1.1831,1.1972,1.1972,1.1972,1.1831,1.1972,1.1972,1.1972,1.1972,
        1.1781,1.1781,1.1918,1.1918,1.1781,1.1781,1.1918,1.1781,1.1918,1.2055,1.2055,1.2055,1.2192,1.2329,1.2329,1.2192,1.2055,1.2055,1.2055,1.1918,1.1918,1.1918,1.1918,1.1918,1.1918,1.1918,1.1918,1.1918,
        1.1867,1.1867,1.1867,1.1867,1.1734,1.1734,1.1867,1.1867,1.2000,1.2000,1.2134,1.2134,1.2267,1.2267,1.2267,1.2267,1.2134,1.2000,1.2000,1.1867,1.2000,1.1867,1.1867,1.1867,1.2000,1.2000,1.1867,1.1867,
        1.1819,1.1819,1.1819,1.1819,1.1689,1.1689,1.1949,1.1819,1.1949,1.1949,1.2078,1.2078,1.2208,1.2338,1.2338,1.2208,1.2078,1.2078,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,
        1.1899,1.1773,1.1899,1.1899,1.1773,1.1773,1.1899,1.1899,1.2026,1.2026,1.2026,1.2152,1.2279,1.2279,1.2279,1.2279,1.2152,1.2026,1.2026,1.1899,1.1899,1.1899,1.1899,1.1899,1.1899,1.1899,1.1899,1.1899,
        1.1852,1.1852,1.1852,1.1852,1.1729,1.1729,1.1852,1.1852,1.1976,1.1976,1.2099,1.2099,1.2223,1.2346,1.2346,1.2223,1.2099,1.1976,1.1976,1.1852,1.1976,1.1976,1.1976,1.1852,1.1976,1.1976,1.1976,1.1976,
        1.1808,1.1808,1.1928,1.1928,1.1808,1.1808,1.1928,1.1808,1.1928,1.2049,1.2049,1.2049,1.2169,1.2290,1.2290,1.2169,1.2049,1.2049,1.2049,1.1928,1.1928,1.1928,1.1928,1.1928,1.1928,1.1928,1.1928,1.1928,
        1.1883,1.1883,1.1883,1.1883,1.1765,1.1765,1.1883,1.1883,1.2000,1.2000,1.2118,1.2118,1.2236,1.2236,1.2236,1.2236,1.2118,1.2000,1.2000,1.1883,1.2000,1.1883,1.1883,1.1883,1.2000,1.2000,1.1883,1.1883,
        1.1840,1.1840,1.1840,1.1840,1.1725,1.1725,1.1955,1.1840,1.1955,1.1955,1.2069,1.2069,1.2184,1.2299,1.2299,1.2184,1.2069,1.2069,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,
        1.1911,1.1798,1.1911,1.1911,1.1798,1.1798,1.1911,1.1911,1.2023,1.2023,1.2023,1.2135,1.2248,1.2248,1.2248,1.2248,1.2135,1.2023,1.2023,1.1911,1.1911,1.1911,1.1911,1.1911,1.1911,1.1911,1.1911,1.1911,
        1.1869,1.1869,1.1869,1.1869,1.1759,1.1759,1.1869,1.1869,1.1979,1.1979,1.2088,1.2088,1.2198,1.2308,1.2198,1.2198,1.2088,1.1979,1.1979,1.1869,1.1979,1.1979,1.1979,1.1869,1.1979,1.1979,1.1979,1.1869,
        1.1828,1.1828,1.1936,1.1936,1.1828,1.1828,1.1936,1.1828,1.1936,1.2044,1.2044,1.2044,1.2151,1.2259,1.2259,1.2151,1.2044,1.2044,1.2044,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,
        1.1895,1.1790,1.1895,1.1895,1.1790,1.1790,1.1895,1.1895,1.2000,1.2000,1.2106,1.2106,1.2211,1.2211,1.2211,1.2211,1.2106,1.2000,1.2000,1.1895,1.2000,1.1895,1.1895,1.1895,1.2000,1.1895,1.1895,1.1895,
        1.1856,1.1856,1.1856,1.1856,1.1753,1.1753,1.1959,1.1856,1.1959,1.1959,1.2062,1.2062,1.2165,1.2269,1.2269,1.2165,1.2062,1.2062,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,
        1.1920,1.1819,1.1920,1.1920,1.1819,1.1819,1.1920,1.1920,1.2021,1.2021,1.2021,1.2122,1.2223,1.2223,1.2223,1.2223,1.2122,1.2021,1.2021,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,
        1.1882,1.1882,1.1882,1.1882,1.1783,1.1783,1.1882,1.1882,1.1981,1.1981,1.2080,1.2080,1.2179,1.2278,1.2179,1.2179,1.2080,1.1981,1.1981,1.1882,1.1981,1.1981,1.1981,1.1882,1.1981,1.1981,1.1882,1.1882,
        1.1845,1.1845,1.1942,1.1942,1.1845,1.1845,1.1942,1.1845,1.1942,1.2039,1.2039,1.2039,1.2136,1.2234,1.2234,1.2136,1.2039,1.2039,1.2039,1.1942,1.1942,1.1942,1.1942,1.1942,1.1942,1.1942,1.1942,1.1942,
        1.1905,1.1810,1.1905,1.1905,1.1810,1.1810,1.1905,1.1905,1.2000,1.2000,1.2096,1.2096,1.2191,1.2191,1.2191,1.2191,1.2096,1.2000,1.2000,1.1905,1.2000,1.1905,1.1905,1.1905,1.2000,1.1905,1.1905,1.1905,
        1.1870,1.1870,1.1870,1.1870,1.1776,1.1776,1.1963,1.1870,1.1963,1.1963,1.2057,1.2057,1.2150,1.2243,1.2243,1.2150,1.2057,1.2057,1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,
        1.1927,1.1835,1.1927,1.1927,1.1835,1.1835,1.1927,1.1927,1.2019,1.2019,1.2019,1.2111,1.2202,1.2202,1.2202,1.2202,1.2111,1.2019,1.2019,1.1927,1.1927,1.1927,1.1927,1.1927,1.1927,1.1927,1.1927,1.1927,
        1.1892,1.1892,1.1892,1.1892,1.1802,1.1802,1.1892,1.1892,1.1982,1.1982,1.2073,1.2073,1.2163,1.2253,1.2163,1.2163,1.2073,1.1982,1.1982,1.1892,1.1982,1.1982,1.1982,1.1892,1.1982,1.1982,1.1892,1.1892,
        1.1859,1.1859,1.1947,1.1947,1.1859,1.1859,1.1947,1.1859,1.1947,1.2036,1.2036,1.2036,1.2124,1.2213,1.2213,1.2124,1.2036,1.2036,1.2036,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,
        1.1914,1.1827,1.1914,1.1914,1.1827,1.1827,1.1914,1.1914,1.2000,1.2000,1.2087,1.2087,1.2174,1.2174,1.2174,1.2174,1.2087,1.2000,1.2000,1.1914,1.2000,1.1914,1.1914,1.1914,1.2000,1.1914,1.1914,1.1914,
        1.1881,1.1881,1.1881,1.1881,1.1795,1.1795,1.1966,1.1881,1.1966,1.1966,1.2052,1.2052,1.2137,1.2223,1.2223,1.2137,1.2052,1.2052,1.1966,1.1966,1.1966,1.1966,1.1966,1.1881,1.1966,1.1966,1.1966,1.1966,
        1.1933,1.1849,1.1933,1.1933,1.1849,1.1849,1.1933,1.1933,1.2017,1.2017,1.2017,1.2101,1.2185,1.2185,1.2185,1.2185,1.2101,1.2017,1.2017,1.1933,1.1933,1.1933,1.1933,1.1933,1.1933,1.1933,1.1933,1.1933,
        1.1901,1.1901,1.1901,1.1901,1.1819,1.1819,1.1901,1.1901,1.1984,1.1984,1.2067,1.2067,1.2149,1.2232,1.2149,1.2149,1.2067,1.1984,1.1984,1.1901,1.1984,1.1984,1.1984,1.1901,1.1984,1.1984,1.1901,1.1901,
        1.1870,1.1870,1.1952,1.1952,1.1870,1.1870,1.1952,1.1870,1.1952,1.2033,1.2033,1.2033,1.2114,1.2196,1.2196,1.2114,1.2033,1.2033,1.2033,1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,
        1.1920,1.1840,1.1920,1.1920,1.1840,1.1840,1.1920,1.1920,1.2000,1.2000,1.2080,1.2080,1.2160,1.2160,1.2160,1.2160,1.2080,1.2000,1.2000,1.1920,1.2000,1.1920,1.1920,1.1920,1.2000,1.1920,1.1920,1.1920,
        1.1890,1.1890,1.1890,1.1890,1.1812,1.1812,1.1969,1.1890,1.1969,1.1969,1.2048,1.2048,1.2126,1.2205,1.2205,1.2126,1.2048,1.2048,1.1969,1.1969,1.1969,1.1969,1.1969,1.1890,1.1969,1.1969,1.1969,1.1969,
        1.1938,1.1861,1.1938,1.1938,1.1861,1.1861,1.1938,1.1938,1.1938,1.2016,1.2016,1.2016,1.2171,1.2171,1.2171,1.2171,1.2016,1.2016,1.2016,1.1938,1.1938,1.1938,1.1938,1.1938,1.1938,1.1938,1.1938,1.1938,
        1.1909,1.1909,1.1909,1.1909,1.1833,1.1833,1.1909,1.1909,1.1985,1.1985,1.2062,1.2062,1.2138,1.2138,1.2138,1.2138,1.2062,1.1985,1.1985,1.1909,1.1985,1.1909,1.1985,1.1909,1.1985,1.1985,1.1909,1.1909,
        1.1880,1.1880,1.1955,1.1955,1.1880,1.1880,1.1955,1.1880,1.1955,1.1955,1.2031,1.2031,1.2106,1.2181,1.2181,1.2106,1.2031,1.2031,1.2031,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,
        1.1926,1.1852,1.1926,1.1926,1.1852,1.1852,1.1926,1.1926,1.2000,1.2000,1.2075,1.2075,1.2149,1.2149,1.2149,1.2149,1.2075,1.2000,1.2000,1.1926,1.1926,1.1926,1.1926,1.1926,1.2000,1.1926,1.1926,1.1926,
        1.1898,1.1898,1.1898,1.1898,1.1825,1.1825,1.1971,1.1898,1.1971,1.1971,1.2044,1.2044,1.2117,1.2190,1.2190,1.2117,1.2044,1.2044,1.1971,1.1971,1.1971,1.1971,1.1971,1.1898,1.1971,1.1971,1.1971,1.1971,
        1.1943,1.1871,1.1943,1.1943,1.1871,1.1871,1.1943,1.1943,1.1943,1.2015,1.2015,1.2015,1.2087,1.2159,1.2159,1.2159,1.2015,1.2015,1.2015,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,
        1.1915,1.1915,1.1915,1.1915,1.1844,1.1844,1.1915,1.1915,1.1986,1.1986,1.2057,1.2057,1.2128,1.2128,1.2128,1.2128,1.2057,1.1986,1.1986,1.1915,1.1986,1.1915,1.1986,1.1915,1.1986,1.1986,1.1915,1.1915,
        1.1889,1.1889,1.1959,1.1959,1.1889,1.1889,1.1959,1.1889,1.1959,1.1959,1.2028,1.2028,1.2098,1.2168,1.2168,1.2098,1.2028,1.2028,1.2028,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,
        1.1932,1.1863,1.1932,1.1932,1.1863,1.1863,1.1932,1.1932,1.2000,1.2000,1.2069,1.2069,1.2138,1.2138,1.2138,1.2138,1.2069,1.2000,1.2000,1.1932,1.1932,1.1932,1.1932,1.1932,1.2000,1.1932,1.1932,1.1932,
        1.1905,1.1905,1.1905,1.1905,1.1837,1.1837,1.1973,1.1905,1.1973,1.1973,1.2041,1.2041,1.2109,1.2177,1.2177,1.2109,1.2041,1.2041,1.1973,1.1973,1.1973,1.1973,1.1973,1.1905,1.1973,1.1973,1.1973,1.1973,
        1.1947,1.1880,1.1947,1.1947,1.1880,1.1880,1.1947,1.1947,1.1947,1.2014,1.2014,1.2014,1.2081,1.2148,1.2148,1.2148,1.2014,1.2014,1.2014,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,
        1.1921,1.1921,1.1921,1.1921,1.1855,1.1855,1.1921,1.1921,1.1987,1.1987,1.2053,1.2053,1.2120,1.2120,1.2120,1.2120,1.2053,1.1987,1.1987,1.1921,1.1987,1.1921,1.1987,1.1921,1.1987,1.1987,1.1921,1.1921,
        1.1896,1.1896,1.1961,1.1961,1.1896,1.1896,1.1961,1.1896,1.1961,1.1961,1.2027,1.2027,1.2092,1.2157,1.2157,1.2092,1.2027,1.2027,1.2027,1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,
        1.1936,1.1871,1.1936,1.1936,1.1871,1.1871,1.1936,1.1936,1.2000,1.2000,1.2000,1.2065,1.2130,1.2130,1.2130,1.2130,1.2065,1.2000,1.2000,1.1936,1.1936,1.1936,1.1936,1.1936,1.2000,1.1936,1.1936,1.1936,
        1.1911,1.1911,1.1911,1.1911,1.1848,1.1848,1.1975,1.1911,1.1975,1.1975,1.2039,1.2039,1.2102,1.2166,1.2166,1.2102,1.2039,1.1975,1.1975,1.1975,1.1975,1.1975,1.1975,1.1911,1.1975,1.1975,1.1975,1.1975,
        1.1950,1.1887,1.1950,1.1950,1.1887,1.1887,1.1950,1.1950,1.1950,1.2013,1.2013,1.2013,1.2076,1.2139,1.2139,1.2076,1.2013,1.2013,1.2013,1.1950,1.1950,1.1950,1.1950,1.1950,1.1950,1.1950,1.1950,1.1950,
        1.1926,1.1926,1.1926,1.1926,1.1864,1.1864,1.1926,1.1926,1.1988,1.1988,1.2050,1.2050,1.2112,1.2112,1.2112,1.2112,1.2050,1.1988,1.1988,1.1926,1.1988,1.1926,1.1988,1.1926,1.1988,1.1988,1.1926,1.1926,
        1.1902,1.1902,1.1964,1.1964,1.1902,1.1902,1.1964,1.1902,1.1964,1.1964,1.2025,1.2025,1.2086,1.2148,1.2148,1.2086,1.2025,1.2025,1.2025,1.1964,1.1964,1.1964,1.1964,1.1964,1.1964,1.1964,1.1964,1.1964,
        1.1940,1.1879,1.1940,1.1940,1.1879,1.1879,1.1940,1.1940,1.2000,1.2000,1.2000,1.2061,1.2122,1.2122,1.2122,1.2122,1.2061,1.2000,1.2000,1.1940,1.1940,1.1940,1.1940,1.1940,1.2000,1.1940,1.1940,1.1940,
        1.1917,1.1917,1.1917,1.1917,1.1857,1.1857,1.1977,1.1917,1.1977,1.1977,1.2036,1.2036,1.2096,1.2156,1.2156,1.2096,1.2036,1.1977,1.1977,1.1977,1.1977,1.1977,1.1977,1.1917,1.1977,1.1977,1.1977,1.1977,
        1.1953,1.1894,1.1953,1.1953,1.1894,1.1894,1.1953,1.1953,1.1953,1.2012,1.2012,1.2012,1.2072,1.2131,1.2131,1.2072,1.2012,1.2012,1.2012,1.1953,1.1953,1.1953,1.1953,1.1953,1.1953,1.1953,1.1953,1.1953,
        1.1930,1.1930,1.1930,1.1930,1.1872,1.1872,1.1930,1.1930,1.1989,1.1989,1.2047,1.2047,1.2106,1.2106,1.2106,1.2106,1.2047,1.1989,1.1989,1.1930,1.1989,1.1930,1.1989,1.1930,1.1989,1.1989,1.1930,1.1930,
        1.1908,1.1908,1.1966,1.1966,1.1908,1.1908,1.1966,1.1908,1.1966,1.1966,1.2024,1.2024,1.2081,1.2139,1.2139,1.2081,1.2024,1.2024,1.2024,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,
        1.1943,1.1886,1.1943,1.1943,1.1886,1.1886,1.1943,1.1943,1.2000,1.2000,1.2000,1.2058,1.2115,1.2115,1.2115,1.2115,1.2058,1.2000,1.2000,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,
        1.1921,1.1921,1.1921,1.1921,1.1865,1.1865,1.1978,1.1921,1.1978,1.1978,1.2034,1.2034,1.2091,1.2147,1.2147,1.2091,1.2034,1.1978,1.1978,1.1978,1.1978,1.1978,1.1978,1.1921,1.1978,1.1978,1.1978,1.1978,
        1.1956,1.1900,1.1956,1.1956,1.1900,1.1900,1.1956,1.1956,1.1956,1.2012,1.2012,1.2012,1.2068,1.2123,1.2123,1.2068,1.2012,1.2012,1.2012,1.1956,1.1956,1.1956,1.1956,1.1956,1.1956,1.1956,1.1956,1.1956,
        1.1934,1.1934,1.1934,1.1934,1.1879,1.1879,1.1934,1.1934,1.1989,1.1989,1.2045,1.2045,1.2100,1.2100,1.2100,1.2100,1.2045,1.1989,1.1989,1.1934,1.1989,1.1934,1.1989,1.1934,1.1989,1.1989,1.1934,1.1934,
        1.1913,1.1913,1.1968,1.1913,1.1913,1.1913,1.1968,1.1913,1.1968,1.1968,1.2022,1.2022,1.2077,1.2132,1.2132,1.2077,1.2022,1.2022,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,
        1.1946,1.1892,1.1946,1.1946,1.1892,1.1892,1.1946,1.1946,1.2000,1.2000,1.2000,1.2055,1.2109,1.2109,1.2109,1.2109,1.2055,1.2000,1.2000,1.1946,1.1946,1.1946,1.1946,1.1946,1.1946,1.1946,1.1946,1.1946,
        1.1926,1.1926,1.1926,1.1926,1.1872,1.1872,1.1979,1.1926,1.1979,1.1979,1.2033,1.2033,1.2086,1.2140,1.2140,1.2086,1.2033,1.1979,1.1979,1.1979,1.1979,1.1979,1.1979,1.1926,1.1979,1.1979,1.1979,1.1979,
        1.1958,1.1905,1.1958,1.1958,1.1905,1.1905,1.1958,1.1958,1.1958,1.2011,1.2011,1.2011,1.2064,1.2117,1.2117,1.2064,1.2011,1.2011,1.2011,1.1958,1.1958,1.1958,1.1958,1.1958,1.1958,1.1958,1.1958,1.1958,
        1.1938,1.1938,1.1938,1.1938,1.1885,1.1885,1.1938,1.1938,1.1990,1.1990,1.2042,1.2042,1.2095,1.2095,1.2095,1.2095,1.2042,1.1990,1.1990,1.1938,1.1990,1.1938,1.1938,1.1938,1.1990,1.1990,1.1938,1.1938,
        1.1918,1.1918,1.1969,1.1918,1.1918,1.1918,1.1969,1.1918,1.1969,1.1969,1.2021,1.2021,1.2073,1.2125,1.2125,1.2073,1.2021,1.2021,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,
        1.1949,1.1898,1.1949,1.1949,1.1898,1.1898,1.1949,1.1949,1.2000,1.2000,1.2000,1.2052,1.2103,1.2103,1.2103,1.2103,1.2052,1.2000,1.2000,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,
        1.1929,1.1929,1.1929,1.1929,1.1879,1.1879,1.1980,1.1929,1.1980,1.1980,1.2031,1.2031,1.2082,1.2132,1.2132,1.2082,1.2031,1.1980,1.1980,1.1980,1.1980,1.1980,1.1980,1.1929,1.1980,1.1980,1.1980,1.1980,
        1.1910,1.1910,1.1960,1.1960,1.1910,1.1910,1.1960,1.1960,1.1960,1.2011,1.2011,1.2011,1.2061,1.2111,1.2111,1.2061,1.2011,1.2011,1.2011,1.1960,1.1960,1.1960,1.1960,1.1960,1.1960,1.1960,1.1960,1.1960,
    ]),

    layer1HFScaleETBins = cms.vint32([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 256]),
    layer1HFScaleFactors = cms.vdouble([
        1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,
        1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,
        1.2000,1.2000,1.0000,1.0000,1.0000,1.0000,1.0000,1.2000,1.0000,1.0000,1.2000,1.2000,
        1.1429,1.1429,1.1429,1.1429,1.1429,1.1429,1.1429,1.1429,1.1429,1.1429,1.1429,1.1429,
        1.1112,1.1112,1.1112,1.1112,1.1112,1.1112,1.1112,1.1112,1.1112,1.1112,1.1112,1.1112,
        1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.0910,1.1819,1.1819,1.1819,
        1.1539,1.1539,1.1539,1.1539,1.1539,1.1539,1.1539,1.1539,1.1539,1.1539,1.1539,1.1539,
        1.2000,1.2000,1.1334,1.1334,1.1334,1.1334,1.1334,1.2000,1.1334,1.1334,1.2000,1.2000,
        1.1765,1.1765,1.1765,1.1765,1.1765,1.1765,1.1765,1.1765,1.1765,1.1765,1.1765,1.1765,
        1.1579,1.1579,1.1579,1.1579,1.1579,1.1579,1.1579,1.1579,1.1579,1.1579,1.1579,1.1579,
        1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1429,1.1905,1.1905,1.1905,
        1.1740,1.1740,1.1740,1.1740,1.1740,1.1740,1.1740,1.1740,1.1740,1.1740,1.1740,1.1740,
        1.2000,1.2000,1.1601,1.1601,1.1601,1.1601,1.1601,1.2000,1.1601,1.1601,1.2000,1.2000,
        1.1852,1.1852,1.1852,1.1852,1.1852,1.1852,1.1852,1.1852,1.1852,1.1852,1.1852,1.1852,
        1.1725,1.1725,1.1725,1.1725,1.1725,1.1725,1.1725,1.1725,1.1725,1.1725,1.1725,1.1725,
        1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1613,1.1936,1.1936,1.1936,
        1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,
        1.2000,1.2000,1.1715,1.1715,1.1715,1.1715,1.1715,1.2000,1.1715,1.1715,1.2000,1.2000,
        1.1892,1.1892,1.1892,1.1892,1.1892,1.1892,1.1892,1.1892,1.1892,1.1892,1.1892,1.1892,
        1.1795,1.1795,1.1795,1.1795,1.1795,1.1795,1.1795,1.1795,1.1795,1.1795,1.1795,1.1795,
        1.1952,1.1952,1.1708,1.1952,1.1952,1.1952,1.1952,1.1952,1.1708,1.1952,1.1952,1.1952,
        1.1861,1.1861,1.1861,1.1861,1.1861,1.1861,1.1861,1.1861,1.1861,1.1861,1.1861,1.1861,
        1.2000,1.2000,1.1778,1.1778,1.1778,1.1778,1.1778,1.2000,1.1778,1.1778,1.2000,1.2000,
        1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,1.1915,
        1.1837,1.1837,1.1837,1.1837,1.1837,1.1837,1.1837,1.1837,1.1837,1.1837,1.1837,1.1837,
        1.1961,1.1961,1.1765,1.1961,1.1961,1.1961,1.1765,1.1961,1.1765,1.1961,1.1961,1.1961,
        1.1887,1.1887,1.1887,1.1887,1.1887,1.1887,1.1887,1.1887,1.1887,1.1887,1.1887,1.1887,
        1.2000,1.1819,1.1819,1.1819,1.1819,1.1819,1.1819,1.2000,1.1819,1.1819,1.2000,1.2000,
        1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,1.1930,
        1.1865,1.1865,1.1865,1.1865,1.1865,1.1865,1.1865,1.1865,1.1865,1.1865,1.1865,1.1865,
        1.1968,1.1968,1.1804,1.1968,1.1968,1.1968,1.1804,1.1968,1.1804,1.1968,1.1968,1.1968,
        1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,
        1.1847,1.1847,1.1847,1.1847,1.1847,1.1847,1.1847,1.2000,1.1847,1.1847,1.2000,1.2000,
        1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,1.1941,
        1.1885,1.1885,1.1885,1.1885,1.1885,1.1885,1.1885,1.1885,1.1885,1.1885,1.1885,1.1885,
        1.1972,1.1972,1.1831,1.1972,1.1972,1.1972,1.1831,1.1972,1.1831,1.1831,1.1972,1.1972,
        1.1918,1.1918,1.1918,1.1918,1.1918,1.1918,1.1918,1.1918,1.1918,1.1918,1.1918,1.1918,
        1.1867,1.1867,1.1867,1.1867,1.1867,1.1867,1.1867,1.1867,1.1867,1.1867,1.2000,1.2000,
        1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,
        1.1899,1.1899,1.1899,1.1899,1.1899,1.1899,1.1899,1.1899,1.1899,1.1899,1.1899,1.1899,
        1.1976,1.1976,1.1852,1.1976,1.1852,1.1976,1.1852,1.1976,1.1852,1.1852,1.1976,1.1976,
        1.1928,1.1928,1.1928,1.1928,1.1928,1.1928,1.1928,1.1928,1.1928,1.1928,1.1928,1.1928,
        1.1883,1.1883,1.1883,1.1883,1.1883,1.1883,1.1883,1.1883,1.1883,1.1883,1.2000,1.2000,
        1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,
        1.1911,1.1911,1.1911,1.1911,1.1911,1.1911,1.1911,1.1911,1.1911,1.1911,1.1911,1.1911,
        1.1979,1.1979,1.1869,1.1979,1.1869,1.1979,1.1869,1.1979,1.1869,1.1869,1.1979,1.1979,
        1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,
        1.1895,1.1895,1.1895,1.1895,1.1895,1.1895,1.1895,1.1895,1.1895,1.1895,1.2000,1.2000,
        1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,
        1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,
        1.1981,1.1981,1.1882,1.1882,1.1882,1.1981,1.1882,1.1981,1.1882,1.1882,1.1981,1.1981,
        1.1942,1.1942,1.1942,1.1942,1.1942,1.1942,1.1942,1.1942,1.1942,1.1942,1.1942,1.1942,
        1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.1905,1.2000,1.1905,
        1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,1.1963,
        1.1927,1.1927,1.1927,1.1927,1.1927,1.1927,1.1927,1.1927,1.1927,1.1927,1.1927,1.1927,
        1.1982,1.1982,1.1892,1.1892,1.1892,1.1982,1.1892,1.1982,1.1892,1.1892,1.1982,1.1982,
        1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,
        1.1914,1.1914,1.1914,1.1914,1.1914,1.1914,1.1914,1.1914,1.1914,1.1914,1.1914,1.1914,
        1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1881,1.1966,1.1966,1.1966,
        1.1933,1.1933,1.1933,1.1933,1.1933,1.1933,1.1933,1.1933,1.1933,1.1933,1.1933,1.1933,
        1.1984,1.1984,1.1901,1.1901,1.1901,1.1984,1.1901,1.1984,1.1901,1.1901,1.1984,1.1984,
        1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,1.1952,
        1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,1.1920,
        1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1890,1.1969,1.1969,1.1969,
        1.1938,1.1938,1.1938,1.1938,1.1938,1.1938,1.1938,1.1938,1.1938,1.1938,1.1938,1.1938,
        1.1985,1.1985,1.1909,1.1909,1.1909,1.1985,1.1909,1.1985,1.1909,1.1909,1.1985,1.1985,
        1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,1.1955,
        1.1926,1.1926,1.1926,1.1926,1.1926,1.1926,1.1926,1.1926,1.1926,1.1926,1.1926,1.1926,
        1.1971,1.1971,1.1971,1.1971,1.1971,1.1971,1.1971,1.1971,1.1898,1.1971,1.1971,1.1971,
        1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,
        1.1986,1.1986,1.1915,1.1915,1.1915,1.1986,1.1915,1.1986,1.1915,1.1915,1.1986,1.1986,
        1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,1.1959,
        1.1932,1.1932,1.1932,1.1932,1.1932,1.1932,1.1932,1.1932,1.1932,1.1932,1.1932,1.1932,
        1.1973,1.1973,1.1973,1.1973,1.1973,1.1973,1.1973,1.1973,1.1905,1.1973,1.1973,1.1973,
        1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,1.1947,
        1.1987,1.1987,1.1921,1.1921,1.1921,1.1921,1.1921,1.1987,1.1921,1.1921,1.1987,1.1987,
        1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,1.1961,
        1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,1.1936,
        1.1975,1.1975,1.1975,1.1975,1.1975,1.1975,1.1975,1.1975,1.1911,1.1975,1.1975,1.1975,
        1.1950,1.1950,1.1950,1.1950,1.1950,1.1950,1.1950,1.1950,1.1950,1.1950,1.1950,1.1950,
        1.1988,1.1988,1.1926,1.1926,1.1926,1.1926,1.1926,1.1988,1.1926,1.1926,1.1988,1.1988,
        1.1964,1.1964,1.1964,1.1964,1.1964,1.1964,1.1964,1.1964,1.1964,1.1964,1.1964,1.1964,
        1.1940,1.1940,1.1940,1.1940,1.1940,1.1940,1.1940,1.1940,1.1940,1.1940,1.1940,1.1940,
        1.1977,1.1977,1.1977,1.1977,1.1977,1.1977,1.1977,1.1977,1.1917,1.1977,1.1977,1.1977,
        1.1953,1.1953,1.1953,1.1953,1.1953,1.1953,1.1953,1.1953,1.1953,1.1953,1.1953,1.1953,
        1.1989,1.1989,1.1930,1.1930,1.1930,1.1930,1.1930,1.1989,1.1930,1.1930,1.1989,1.1989,
        1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,1.1966,
        1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,1.1943,
        1.1978,1.1978,1.1978,1.1978,1.1978,1.1978,1.1978,1.1978,1.1921,1.1978,1.1978,1.1978,
        1.1956,1.1956,1.1956,1.1956,1.1956,1.1956,1.1956,1.1956,1.1956,1.1956,1.1956,1.1956,
        1.1989,1.1989,1.1934,1.1934,1.1934,1.1934,1.1934,1.1989,1.1934,1.1934,1.1989,1.1989,
        1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,1.1968,
        1.1946,1.1946,1.1946,1.1946,1.1946,1.1946,1.1946,1.1946,1.1946,1.1946,1.1946,1.1946,
        1.1979,1.1979,1.1926,1.1979,1.1979,1.1979,1.1979,1.1979,1.1926,1.1979,1.1979,1.1979,
        1.1958,1.1958,1.1958,1.1958,1.1958,1.1958,1.1958,1.1958,1.1958,1.1958,1.1958,1.1958,
        1.1990,1.1990,1.1938,1.1938,1.1938,1.1938,1.1938,1.1990,1.1938,1.1938,1.1990,1.1990,
        1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,1.1969,
        1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,1.1949,
        1.1980,1.1980,1.1929,1.1980,1.1980,1.1980,1.1929,1.1980,1.1929,1.1980,1.1980,1.1980,
        1.1960,1.1960,1.1960,1.1960,1.1960,1.1960,1.1960,1.1960,1.1960,1.1960,1.1960,1.1960,
    ]),

    # HCal FB LUT
    layer1HCalFBLUTUpper = cms.vuint32([
    0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 
    ]),

    layer1HCalFBLUTLower = cms.vuint32([
    0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 0xBBBABBBA, 
    ])
)