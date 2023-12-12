# start first bin from -1 so that 0 gets included in it
# this is from caloParams_2021_v0_6_cfi
layer1ECalScaleETBins_currCalib = [-1, 3, 6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256]
layer1ECalScaleETLabels_currCalib = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
layer1ECalScaleFactors_currCalib = [1.12, 1.13, 1.13, 1.12, 1.12, 1.12, 1.13, 1.12, 1.13, 1.12, 1.13, 1.13, 1.14, 1.13, 1.13, 1.13, 1.14, 1.26, 1.11, 1.20, 1.21, 1.22, 1.19, 1.20, 1.19, 0.00, 0.00, 0.00,
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
                                   1.02, 1.00, 1.00, 1.02, 1.00, 1.01, 1.01, 1.00, 1.00, 1.02, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.02, 1.04, 1.01, 1.03, 1.03, 1.03, 1.03, 1.02, 1.02, 1.02, 1.00, 1.01]


# start first bin from -1 so that 0 gets included in it
# this is the fusion of the HCAL and HF parameters from the latest working had calibration caloParams_2021_v0_2_cfi
layer1HCalScaleETBins_oldCalib = [-1, 6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256]
layer1HCalScaleETLabels_oldCalib = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
layer1HCalScaleFactors_oldCalib = [1.55, 1.59, 1.60, 1.60, 1.58, 1.62, 1.63, 1.63, 1.63, 1.65, 1.65, 1.71, 1.69, 1.72, 1.84, 1.98, 1.98, 1.51, 1.55, 1.56, 1.42, 1.44, 1.46, 1.46, 1.51, 1.44, 1.29, 1.23, 1.35, 1.09, 1.12, 1.10, 1.17, 1.18, 1.19, 1.23, 1.25, 1.32, 1.61, 1.79,
                                   1.39, 1.39, 1.40, 1.42, 1.40, 1.42, 1.45, 1.43, 1.43, 1.45, 1.47, 1.49, 1.47, 1.51, 1.57, 1.67, 1.70, 1.32, 1.35, 1.36, 1.24, 1.26, 1.27, 1.30, 1.32, 1.31, 1.16, 1.10, 1.27, 1.01, 1.09, 1.03, 1.04, 1.05, 1.09, 1.11, 1.18, 1.19, 1.48, 1.67,
                                   1.31, 1.33, 1.33, 1.34, 1.33, 1.34, 1.35, 1.37, 1.36, 1.37, 1.39, 1.39, 1.39, 1.39, 1.45, 1.54, 1.57, 1.22, 1.25, 1.27, 1.16, 1.19, 1.20, 1.22, 1.25, 1.24, 1.10, 1.05, 1.15, 0.98, 1.05, 1.02, 1.00, 0.99, 1.03, 1.04, 1.10, 1.12, 1.39, 1.66,
                                   1.27, 1.28, 1.29, 1.29, 1.29, 1.28, 1.31, 1.31, 1.30, 1.31, 1.33, 1.34, 1.33, 1.34, 1.41, 1.46, 1.48, 1.19, 1.20, 1.20, 1.12, 1.13, 1.15, 1.17, 1.20, 1.20, 1.06, 1.01, 1.14, 0.96, 1.03, 0.97, 0.96, 0.96, 0.98, 1.00, 1.04, 1.07, 1.35, 1.59,
                                   1.22, 1.22, 1.23, 1.23, 1.23, 1.24, 1.24, 1.26, 1.25, 1.27, 1.27, 1.28, 1.28, 1.27, 1.32, 1.38, 1.41, 1.12, 1.15, 1.16, 1.08, 1.10, 1.11, 1.13, 1.15, 1.15, 1.03, 0.98, 1.07, 0.97, 1.00, 0.96, 0.91, 0.92, 0.95, 0.96, 1.01, 1.03, 1.28, 1.56,
                                   1.17, 1.19, 1.17, 1.19, 1.19, 1.19, 1.20, 1.22, 1.20, 1.21, 1.21, 1.22, 1.22, 1.23, 1.26, 1.31, 1.33, 1.10, 1.10, 1.10, 1.04, 1.06, 1.07, 1.09, 1.11, 1.10, 0.99, 0.95, 1.03, 0.94, 0.97, 0.94, 0.88, 0.90, 0.92, 0.94, 0.98, 1.01, 1.27, 1.53,
                                   1.14, 1.15, 1.14, 1.15, 1.16, 1.15, 1.16, 1.17, 1.16, 1.17, 1.19, 1.18, 1.18, 1.19, 1.22, 1.26, 1.26, 1.06, 1.07, 1.08, 1.02, 1.03, 1.04, 1.06, 1.07, 1.07, 0.96, 0.92, 1.01, 0.92, 0.96, 0.90, 0.87, 0.89, 0.91, 0.93, 0.96, 0.99, 1.23, 1.48,
                                   1.11, 1.11, 1.13, 1.12, 1.11, 1.13, 1.13, 1.13, 1.12, 1.14, 1.15, 1.15, 1.14, 1.15, 1.17, 1.20, 1.23, 1.03, 1.05, 1.05, 1.00, 1.01, 1.02, 1.03, 1.05, 1.03, 0.95, 0.91, 0.98, 0.89, 0.96, 0.87, 0.86, 0.87, 0.89, 0.91, 0.94, 0.97, 1.19, 1.47,
                                   1.08, 1.09, 1.09, 1.08, 1.09, 1.10, 1.10, 1.11, 1.11, 1.11, 1.12, 1.11, 1.11, 1.12, 1.13, 1.17, 1.16, 1.01, 1.02, 1.03, 0.98, 0.99, 0.99, 1.01, 1.02, 1.01, 0.94, 0.89, 0.95, 0.88, 0.94, 0.87, 0.86, 0.86, 0.88, 0.90, 0.94, 0.96, 1.16, 1.43,
                                   1.06, 1.07, 1.06, 1.07, 1.07, 1.07, 1.08, 1.08, 1.07, 1.07, 1.08, 1.08, 1.08, 1.09, 1.10, 1.14, 1.13, 1.00, 1.02, 1.02, 0.97, 0.98, 0.98, 0.99, 1.00, 1.00, 0.92, 0.87, 0.93, 0.88, 0.93, 0.87, 0.86, 0.87, 0.88, 0.90, 0.93, 0.95, 1.14, 1.42,
                                   1.03, 1.04, 1.04, 1.04, 1.04, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.06, 1.06, 1.09, 1.09, 0.97, 0.99, 1.00, 0.95, 0.96, 0.96, 0.97, 0.99, 0.98, 0.90, 0.85, 0.92, 0.86, 0.90, 0.86, 0.85, 0.86, 0.88, 0.89, 0.92, 0.95, 1.12, 1.41,
                                   1.00, 1.00, 1.00, 1.01, 1.01, 1.01, 1.02, 1.02, 1.01, 1.01, 1.02, 1.02, 1.01, 0.98, 1.01, 1.02, 1.02, 0.96, 0.97, 1.00, 0.93, 0.94, 0.94, 0.95, 0.96, 0.96, 0.89, 0.82, 0.90, 0.85, 0.90, 0.85, 0.84, 0.86, 0.88, 0.90, 0.93, 0.95, 1.09, 1.35,
                                   0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.95, 0.95, 0.95, 0.96, 0.95, 0.93, 0.95, 0.95, 0.93, 0.93, 0.94, 0.94, 0.95, 0.95, 0.88, 0.82, 0.86, 0.85, 0.89, 0.85, 0.85, 0.86, 0.88, 0.90, 0.93, 0.95, 1.10, 1.27]

# start first bin from -1 so that 0 gets included in it
# this is the fusion of the HCAL and HF parameters from the latest working had calibration caloParams_2021_v0_6_cfi
layer1HCalScaleETBins_currCalib = [-1, 6, 9, 12, 15, 20, 25, 30, 35, 40, 45, 55, 70, 256]
layer1HCalScaleETLabels_currCalib = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
layer1HCalScaleFactors_currCalib = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 
                                    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

# Add ECAL new SF
layer1ECalScaleETBins_v33_newCalib = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 256]
layer1ECalScaleETLabels_v33_newCalib = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
layer1ECalScaleFactors_v33_newCalib = [ 0.3333,0.3333,0.3333,0.3333,0.3333,0.3333,0.3333,0.3333,0.3333,0.6667,0.6667,0.6667,0.6667,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,0.6667,0.6667,0.6667,0.0000,0.0000,0.0000,
        0.6000,0.6000,0.6000,0.6000,0.6000,0.6000,0.8000,0.8000,0.8000,0.8000,0.8000,0.8000,1.0000,1.0000,1.0000,1.0000,1.0000,1.2000,1.0000,1.2000,1.2000,1.0000,1.0000,1.0000,1.0000,0.0000,0.0000,0.0000,
        0.7143,0.7143,0.7143,0.8571,0.8571,0.7143,0.8571,0.8571,0.8571,0.8571,0.8571,1.0000,1.0000,1.1429,1.1429,1.1429,1.1429,1.2857,1.1429,1.2857,1.1429,1.1429,1.1429,1.0000,1.0000,0.7143,0.0000,0.0000,
        0.8889,0.8889,0.8889,0.8889,0.8889,0.8889,0.8889,0.8889,1.0000,1.0000,1.0000,1.0000,1.1111,1.1111,1.1111,1.2222,1.2222,1.2222,1.2222,1.2222,1.2222,1.2222,1.1111,1.1111,1.0000,0.8889,0.0000,0.0000,
        0.9091,0.9091,0.9091,0.9091,0.9091,0.9091,0.9091,1.0000,1.0000,1.0000,1.0000,1.0909,1.0909,1.1818,1.1818,1.1818,1.1818,1.2727,1.1818,1.2727,1.2727,1.1818,1.1818,1.0909,1.0909,1.0000,0.0000,0.0000,
        1.0000,0.9231,0.9231,1.0000,1.0000,1.0000,1.0000,1.0000,1.0000,1.0769,1.0000,1.0769,1.1538,1.1538,1.1538,1.1538,1.2308,1.3077,1.2308,1.3077,1.3077,1.2308,1.2308,1.1538,1.1538,1.0000,1.0769,0.0000,
        1.0000,1.0000,1.0000,1.0000,1.0000,1.0667,1.0000,1.0000,1.0000,1.0667,1.0667,1.0667,1.1333,1.2000,1.2000,1.2000,1.2000,1.2667,1.2000,1.2667,1.2667,1.2667,1.2000,1.2000,1.1333,1.0667,1.1333,0.0000,
        1.0000,1.0000,1.0000,1.0000,1.0588,1.0588,1.0588,1.0000,1.0588,1.0588,1.0588,1.0588,1.1176,1.1765,1.1765,1.1765,1.1765,1.2941,1.1765,1.2941,1.2941,1.2353,1.2353,1.1765,1.1765,1.1176,1.1176,0.0000,
        1.0526,1.0000,1.0000,1.0000,1.0526,1.0526,1.0526,1.0526,1.0526,1.1053,1.0526,1.1053,1.1579,1.2105,1.1579,1.2105,1.2105,1.2632,1.2105,1.2632,1.2632,1.2632,1.2105,1.2105,1.1579,1.1579,1.1579,0.0000,
        1.0476,1.0476,1.0476,1.0476,1.0476,1.0476,1.0476,1.0476,1.0476,1.0952,1.0476,1.0952,1.1429,1.1905,1.1905,1.1905,1.1905,1.2857,1.1905,1.2857,1.2381,1.2381,1.2381,1.1905,1.1905,1.1429,1.1429,0.0000,
        1.0435,1.0435,1.0435,1.0435,1.0435,1.0435,1.0435,1.0435,1.0435,1.0870,1.0870,1.0870,1.1304,1.1739,1.1739,1.1739,1.1739,1.2609,1.1739,1.2609,1.2609,1.2609,1.2174,1.2174,1.1739,1.1739,1.1739,0.0000,
        1.0400,1.0400,1.0400,1.0400,1.0400,1.0400,1.0400,1.0400,1.0800,1.0800,1.0800,1.0800,1.1600,1.2000,1.1600,1.1600,1.2000,1.2800,1.1600,1.2400,1.2400,1.2800,1.2000,1.2000,1.2000,1.1600,1.1600,0.0000,
        1.0370,1.0370,1.0370,1.0370,1.0370,1.0741,1.0741,1.0370,1.0741,1.0741,1.0741,1.1111,1.1481,1.1852,1.1852,1.1852,1.1852,1.2593,1.1852,1.2593,1.2593,1.2593,1.2222,1.2222,1.1852,1.1852,1.1481,0.0000,
        1.0345,1.0345,1.0345,1.0345,1.0345,1.0690,1.0690,1.0690,1.0690,1.0690,1.0690,1.1034,1.1379,1.1724,1.1724,1.1724,1.1724,1.2414,1.1724,1.2414,1.2414,1.2414,1.2069,1.2069,1.2069,1.1724,1.1724,0.0000,
        1.0323,1.0323,1.0323,1.0323,1.0323,1.0645,1.0645,1.0645,1.0645,1.0645,1.0968,1.0968,1.1290,1.1613,1.1613,1.1613,1.1935,1.2581,1.1613,1.2258,1.2258,1.2258,1.2258,1.2258,1.1935,1.1935,1.1613,0.0000,
        1.0303,1.0303,1.0303,1.0303,1.0303,1.0606,1.0606,1.0606,1.0606,1.0909,1.0909,1.0909,1.1212,1.1515,1.1515,1.1515,1.1818,1.2424,1.1515,1.2424,1.2424,1.2424,1.2121,1.2121,1.1818,1.1818,1.1515,0.0000,
        1.0286,1.0286,1.0286,1.0286,1.0286,1.0571,1.0571,1.0571,1.0571,1.0857,1.0857,1.0857,1.1143,1.1429,1.1429,1.1714,1.1714,1.2571,1.1714,1.2286,1.2286,1.2286,1.2000,1.2000,1.2000,1.1714,1.1429,0.0000,
        1.0270,1.0270,1.0270,1.0270,1.0270,1.0541,1.0541,1.0541,1.0541,1.0811,1.0811,1.0811,1.1081,1.1622,1.1351,1.1622,1.1622,1.2432,1.1622,1.2162,1.2162,1.2162,1.1892,1.1892,1.1892,1.1892,1.1351,0.0000,
        1.0256,1.0256,1.0256,1.0256,1.0256,1.0513,1.0513,1.0513,1.0513,1.0769,1.0769,1.0769,1.1026,1.1538,1.1538,1.1538,1.1538,1.2308,1.1538,1.2308,1.2308,1.2051,1.2051,1.2051,1.1795,1.1795,1.1282,0.0000,
        1.0244,1.0244,1.0244,1.0244,1.0244,1.0488,1.0488,1.0488,1.0488,1.0732,1.0732,1.0732,1.0976,1.1463,1.1463,1.1463,1.1463,1.2195,1.1463,1.2195,1.2195,1.2195,1.1951,1.1951,1.1707,1.1707,1.1220,0.0244,
        1.0233,1.0233,1.0233,1.0233,1.0233,1.0465,1.0465,1.0465,1.0465,1.0698,1.0698,1.0698,1.0930,1.1395,1.1395,1.1395,1.1395,1.2326,1.1395,1.2093,1.2093,1.2093,1.1860,1.1860,1.1628,1.1860,1.1163,0.0465,
        1.0222,1.0222,1.0222,1.0222,1.0222,1.0444,1.0444,1.0444,1.0667,1.0667,1.0667,1.0667,1.1111,1.1333,1.1333,1.1333,1.1333,1.2222,1.1556,1.2000,1.2000,1.2000,1.1778,1.1778,1.1778,1.1778,1.1111,0.0667,
        1.0213,1.0213,1.0213,1.0213,1.0426,1.0426,1.0426,1.0426,1.0638,1.0638,1.0638,1.0638,1.1064,1.1277,1.1277,1.1277,1.1277,1.2128,1.1489,1.2128,1.1915,1.1915,1.1915,1.1915,1.1702,1.1702,1.1064,0.0851,
        1.0204,1.0204,1.0204,1.0204,1.0408,1.0408,1.0408,1.0408,1.0612,1.0612,1.0612,1.0816,1.1020,1.1224,1.1224,1.1224,1.1429,1.2041,1.1429,1.2041,1.2041,1.1837,1.1837,1.1837,1.1633,1.1633,1.1020,0.1224,
        1.0196,1.0196,1.0196,1.0196,1.0392,1.0392,1.0392,1.0392,1.0588,1.0588,1.0588,1.0784,1.0980,1.1176,1.1176,1.1176,1.1373,1.1961,1.1373,1.1961,1.1961,1.1961,1.1765,1.1765,1.1569,1.1765,1.0980,0.1373,
        1.0189,1.0189,1.0189,1.0189,1.0377,1.0377,1.0377,1.0377,1.0566,1.0566,1.0566,1.0755,1.0943,1.1132,1.1132,1.1132,1.1321,1.1887,1.1321,1.1887,1.1887,1.1887,1.1698,1.1698,1.1509,1.1698,1.0943,0.1509,
        1.0182,1.0182,1.0182,1.0182,1.0364,1.0364,1.0364,1.0364,1.0545,1.0545,1.0545,1.0727,1.0909,1.1091,1.1091,1.1091,1.1273,1.1818,1.1273,1.1818,1.1818,1.1818,1.1636,1.1636,1.1455,1.1636,1.0909,0.1818,
        1.0175,1.0175,1.0175,1.0175,1.0351,1.0351,1.0351,1.0351,1.0526,1.0526,1.0526,1.0702,1.0877,1.1053,1.1053,1.1053,1.1228,1.1754,1.1228,1.1930,1.1754,1.1754,1.1579,1.1579,1.1404,1.1579,1.0877,0.2105,
        1.0169,1.0169,1.0169,1.0169,1.0339,1.0339,1.0339,1.0339,1.0508,1.0508,1.0508,1.0678,1.0847,1.1017,1.1017,1.1186,1.1186,1.1695,1.1186,1.1864,1.1695,1.1695,1.1525,1.1525,1.1356,1.1525,1.0847,0.2373,
        1.0164,1.0164,1.0164,1.0164,1.0328,1.0328,1.0328,1.0328,1.0492,1.0492,1.0492,1.0656,1.0820,1.0984,1.0984,1.1148,1.1148,1.1639,1.1148,1.1803,1.1639,1.1639,1.1475,1.1475,1.1311,1.1475,1.0820,0.2623,
        1.0159,1.0159,1.0159,1.0159,1.0317,1.0317,1.0317,1.0317,1.0476,1.0476,1.0476,1.0635,1.0794,1.0952,1.0952,1.1111,1.1111,1.1587,1.1111,1.1746,1.1587,1.1587,1.1587,1.1429,1.1429,1.1429,1.0794,0.2857,
        1.0154,1.0154,1.0154,1.0308,1.0308,1.0308,1.0308,1.0308,1.0462,1.0462,1.0462,1.0615,1.0769,1.0923,1.0923,1.1077,1.1077,1.1538,1.1077,1.1692,1.1538,1.1538,1.1538,1.1385,1.1385,1.1385,1.0769,0.3077,
        1.0149,1.0149,1.0149,1.0299,1.0299,1.0299,1.0299,1.0299,1.0448,1.0448,1.0448,1.0597,1.0746,1.0896,1.0896,1.1045,1.1045,1.1493,1.1045,1.1642,1.1493,1.1493,1.1493,1.1493,1.1343,1.1343,1.0746,0.3284,
        1.0145,1.0145,1.0145,1.0290,1.0290,1.0290,1.0290,1.0290,1.0435,1.0435,1.0435,1.0580,1.0725,1.0870,1.0870,1.1014,1.1014,1.1449,1.1014,1.1594,1.1449,1.1449,1.1449,1.1449,1.1304,1.1304,1.0870,0.3478,
        1.0141,1.0141,1.0141,1.0282,1.0282,1.0282,1.0282,1.0282,1.0423,1.0423,1.0423,1.0563,1.0704,1.0845,1.0845,1.0986,1.0986,1.1408,1.0986,1.1549,1.1408,1.1408,1.1408,1.1408,1.1268,1.1268,1.0845,0.3662,
        1.0137,1.0137,1.0137,1.0274,1.0274,1.0274,1.0274,1.0274,1.0411,1.0411,1.0411,1.0548,1.0685,1.0822,1.0822,1.0959,1.0959,1.1370,1.0959,1.1507,1.1370,1.1370,1.1370,1.1370,1.1233,1.1370,1.0822,0.3973,
        1.0133,1.0133,1.0133,1.0267,1.0267,1.0267,1.0267,1.0267,1.0400,1.0400,1.0400,1.0533,1.0667,1.0800,1.0800,1.0933,1.0933,1.1333,1.0933,1.1467,1.1333,1.1333,1.1333,1.1333,1.1200,1.1333,1.0800,0.4133,
        1.0130,1.0130,1.0130,1.0260,1.0260,1.0260,1.0260,1.0260,1.0390,1.0390,1.0390,1.0519,1.0649,1.0779,1.0779,1.0909,1.0909,1.1299,1.0909,1.1429,1.1299,1.1299,1.1299,1.1299,1.1169,1.1299,1.0779,0.4286,
        1.0127,1.0127,1.0127,1.0253,1.0253,1.0253,1.0253,1.0253,1.0380,1.0380,1.0380,1.0506,1.0633,1.0886,1.0886,1.0886,1.0886,1.1392,1.0886,1.1392,1.1266,1.1266,1.1266,1.1266,1.1139,1.1266,1.0759,0.4430,
        1.0123,1.0123,1.0123,1.0247,1.0247,1.0247,1.0247,1.0247,1.0370,1.0370,1.0370,1.0494,1.0617,1.0864,1.0864,1.0864,1.0864,1.1358,1.0864,1.1358,1.1235,1.1235,1.1235,1.1235,1.1111,1.1235,1.0741,0.4691,
        1.0120,1.0120,1.0120,1.0241,1.0241,1.0241,1.0241,1.0241,1.0361,1.0361,1.0361,1.0482,1.0602,1.0843,1.0843,1.0843,1.0843,1.1325,1.0843,1.1325,1.1325,1.1205,1.1205,1.1205,1.1084,1.1205,1.0723,0.4819,
        1.0118,1.0118,1.0118,1.0235,1.0235,1.0235,1.0235,1.0235,1.0353,1.0353,1.0353,1.0471,1.0588,1.0824,1.0824,1.0824,1.0824,1.1294,1.0824,1.1294,1.1294,1.1176,1.1176,1.1176,1.1059,1.1176,1.0706,0.4941,
        1.0115,1.0115,1.0115,1.0230,1.0230,1.0230,1.0230,1.0230,1.0345,1.0345,1.0345,1.0460,1.0575,1.0805,1.0805,1.0805,1.0805,1.1264,1.0805,1.1264,1.1264,1.1149,1.1149,1.1149,1.1034,1.1149,1.0690,0.5057,
        1.0112,1.0112,1.0112,1.0225,1.0225,1.0225,1.0225,1.0225,1.0337,1.0337,1.0337,1.0449,1.0562,1.0787,1.0787,1.0787,1.0787,1.1236,1.0899,1.1236,1.1236,1.1236,1.1124,1.1124,1.1011,1.1124,1.0674,0.5169,
        1.0110,1.0110,1.0110,1.0220,1.0220,1.0220,1.0220,1.0220,1.0330,1.0330,1.0330,1.0440,1.0549,1.0769,1.0769,1.0769,1.0769,1.1209,1.0879,1.1209,1.1209,1.1209,1.1099,1.1099,1.0989,1.1099,1.0659,0.5275,
        1.0108,1.0108,1.0108,1.0215,1.0215,1.0215,1.0215,1.0215,1.0323,1.0323,1.0323,1.0430,1.0645,1.0753,1.0753,1.0753,1.0753,1.1183,1.0860,1.1183,1.1183,1.1183,1.1075,1.1075,1.0968,1.1075,1.0645,0.5376,
        1.0105,1.0211,1.0105,1.0211,1.0211,1.0211,1.0211,1.0211,1.0316,1.0316,1.0316,1.0421,1.0632,1.0737,1.0737,1.0737,1.0842,1.1158,1.0842,1.1158,1.1158,1.1158,1.1053,1.1053,1.0947,1.1053,1.0632,0.5474,
        1.0206,1.0206,1.0103,1.0206,1.0206,1.0206,1.0206,1.0206,1.0309,1.0309,1.0309,1.0515,1.0619,1.0722,1.0722,1.0722,1.0825,1.1134,1.0825,1.1134,1.1134,1.1134,1.1031,1.1031,1.0928,1.1031,1.0619,0.5567,
        1.0202,1.0202,1.0101,1.0202,1.0202,1.0202,1.0202,1.0202,1.0303,1.0303,1.0303,1.0505,1.0606,1.0707,1.0707,1.0707,1.0808,1.1111,1.0808,1.1111,1.1111,1.1111,1.1010,1.1010,1.0909,1.1010,1.0606,0.5657,
        1.0198,1.0198,1.0099,1.0198,1.0198,1.0198,1.0198,1.0198,1.0297,1.0297,1.0297,1.0495,1.0594,1.0693,1.0693,1.0693,1.0792,1.1089,1.0792,1.1089,1.1089,1.1089,1.0990,1.0990,1.0891,1.0990,1.0594,0.5743,
        1.0194,1.0194,1.0097,1.0194,1.0194,1.0194,1.0194,1.0194,1.0291,1.0291,1.0291,1.0485,1.0583,1.0680,1.0680,1.0680,1.0777,1.1068,1.0777,1.1165,1.1068,1.1068,1.0971,1.0971,1.0874,1.0971,1.0583,0.5825,
        1.0190,1.0190,1.0095,1.0190,1.0190,1.0190,1.0190,1.0190,1.0286,1.0286,1.0286,1.0476,1.0571,1.0667,1.0667,1.0667,1.0762,1.1048,1.0762,1.1143,1.1048,1.1048,1.0952,1.0952,1.0857,1.0952,1.0571,0.5905,
        1.0187,1.0187,1.0093,1.0187,1.0187,1.0187,1.0187,1.0187,1.0280,1.0280,1.0280,1.0467,1.0561,1.0654,1.0654,1.0748,1.0748,1.1028,1.0748,1.1121,1.1028,1.1028,1.1028,1.0935,1.0935,1.0935,1.0561,0.5981,
        1.0183,1.0183,1.0092,1.0183,1.0183,1.0183,1.0183,1.0183,1.0275,1.0275,1.0275,1.0459,1.0550,1.0642,1.0642,1.0734,1.0734,1.1009,1.0734,1.1101,1.1009,1.1009,1.1009,1.0917,1.0917,1.0917,1.0550,0.6055,
        1.0180,1.0180,1.0090,1.0180,1.0180,1.0180,1.0180,1.0180,1.0270,1.0270,1.0270,1.0450,1.0541,1.0631,1.0631,1.0721,1.0721,1.0991,1.0721,1.1081,1.0991,1.0991,1.0991,1.0901,1.0901,1.0901,1.0541,0.6126,
        1.0177,1.0177,1.0088,1.0177,1.0177,1.0177,1.0177,1.0177,1.0265,1.0265,1.0265,1.0442,1.0531,1.0619,1.0619,1.0708,1.0708,1.0973,1.0708,1.1062,1.0973,1.0973,1.0973,1.0973,1.0885,1.0885,1.0531,0.6195,
        1.0174,1.0174,1.0087,1.0174,1.0174,1.0174,1.0174,1.0174,1.0261,1.0261,1.0261,1.0435,1.0522,1.0609,1.0609,1.0696,1.0696,1.0957,1.0696,1.1043,1.0957,1.0957,1.0957,1.0957,1.0870,1.0870,1.0609,0.6261,
        1.0171,1.0171,1.0085,1.0171,1.0171,1.0171,1.0171,1.0171,1.0256,1.0256,1.0256,1.0427,1.0513,1.0598,1.0598,1.0684,1.0684,1.0940,1.0684,1.1026,1.0940,1.0940,1.0940,1.0940,1.0855,1.0855,1.0598,0.6325,
        1.0168,1.0168,1.0084,1.0168,1.0168,1.0168,1.0168,1.0168,1.0252,1.0252,1.0252,1.0420,1.0504,1.0588,1.0588,1.0672,1.0672,1.0924,1.0672,1.1008,1.0924,1.0924,1.0924,1.0924,1.0840,1.0840,1.0588,0.6387,
        1.0165,1.0165,1.0083,1.0165,1.0165,1.0165,1.0165,1.0165,1.0248,1.0248,1.0248,1.0413,1.0496,1.0579,1.0579,1.0661,1.0661,1.0909,1.0661,1.0992,1.0909,1.0909,1.0909,1.0909,1.0826,1.0909,1.0579,0.6446,
        1.0163,1.0163,1.0081,1.0163,1.0163,1.0163,1.0163,1.0163,1.0244,1.0244,1.0244,1.0407,1.0488,1.0569,1.0569,1.0650,1.0650,1.0894,1.0650,1.0976,1.0894,1.0894,1.0894,1.0894,1.0813,1.0894,1.0569,0.6504,
        1.0160,1.0160,1.0080,1.0160,1.0160,1.0160,1.0160,1.0160,1.0240,1.0240,1.0240,1.0400,1.0480,1.0560,1.0560,1.0640,1.0640,1.0880,1.0640,1.0960,1.0880,1.0880,1.0880,1.0880,1.0800,1.0880,1.0560,0.6560,
        1.0157,1.0157,1.0079,1.0157,1.0157,1.0157,1.0157,1.0157,1.0236,1.0236,1.0236,1.0394,1.0472,1.0630,1.0630,1.0630,1.0630,1.0945,1.0630,1.0945,1.0866,1.0866,1.0866,1.0866,1.0787,1.0866,1.0551,0.6614,
        1.0155,1.0155,1.0078,1.0155,1.0155,1.0155,1.0155,1.0155,1.0233,1.0233,1.0233,1.0388,1.0465,1.0620,1.0620,1.0620,1.0620,1.0930,1.0620,1.0930,1.0853,1.0853,1.0853,1.0853,1.0775,1.0853,1.0543,0.6667,
        1.0153,1.0153,1.0076,1.0153,1.0153,1.0153,1.0153,1.0153,1.0229,1.0229,1.0229,1.0382,1.0458,1.0611,1.0611,1.0611,1.0611,1.0916,1.0611,1.0916,1.0916,1.0840,1.0840,1.0840,1.0763,1.0840,1.0534,0.6794,
        1.0150,1.0150,1.0150,1.0150,1.0150,1.0150,1.0150,1.0150,1.0226,1.0226,1.0226,1.0376,1.0451,1.0602,1.0602,1.0602,1.0602,1.0902,1.0602,1.0902,1.0902,1.0827,1.0827,1.0827,1.0752,1.0827,1.0526,0.6842,
        1.0148,1.0148,1.0148,1.0148,1.0148,1.0148,1.0148,1.0148,1.0222,1.0222,1.0222,1.0370,1.0444,1.0593,1.0593,1.0593,1.0593,1.0889,1.0667,1.0889,1.0889,1.0815,1.0815,1.0815,1.0741,1.0815,1.0519,0.6889,
        1.0146,1.0146,1.0146,1.0146,1.0146,1.0146,1.0146,1.0146,1.0219,1.0219,1.0219,1.0365,1.0438,1.0584,1.0584,1.0584,1.0584,1.0876,1.0657,1.0876,1.0876,1.0876,1.0803,1.0803,1.0730,1.0803,1.0511,0.6934,
        1.0144,1.0144,1.0144,1.0144,1.0144,1.0144,1.0144,1.0144,1.0216,1.0216,1.0216,1.0360,1.0432,1.0576,1.0576,1.0576,1.0576,1.0863,1.0647,1.0863,1.0863,1.0863,1.0791,1.0791,1.0719,1.0791,1.0504,0.6978,
        1.0142,1.0142,1.0142,1.0142,1.0142,1.0142,1.0142,1.0142,1.0213,1.0213,1.0213,1.0355,1.0496,1.0567,1.0567,1.0567,1.0567,1.0851,1.0638,1.0851,1.0851,1.0851,1.0780,1.0780,1.0709,1.0780,1.0496,0.7021,
        1.0140,1.0140,1.0140,1.0140,1.0140,1.0140,1.0140,1.0140,1.0210,1.0210,1.0210,1.0420,1.0490,1.0559,1.0559,1.0559,1.0629,1.0839,1.0629,1.0839,1.0839,1.0839,1.0769,1.0769,1.0699,1.0769,1.0490,0.7063,
        1.0138,1.0138,1.0138,1.0138,1.0138,1.0138,1.0138,1.0138,1.0207,1.0207,1.0207,1.0414,1.0483,1.0552,1.0552,1.0552,1.0621,1.0828,1.0621,1.0828,1.0828,1.0828,1.0759,1.0759,1.0690,1.0759,1.0483,0.7103,
        1.0136,1.0136,1.0136,1.0136,1.0136,1.0136,1.0136,1.0136,1.0204,1.0204,1.0204,1.0408,1.0476,1.0544,1.0544,1.0544,1.0612,1.0816,1.0612,1.0816,1.0816,1.0816,1.0748,1.0748,1.0680,1.0748,1.0476,0.7143,
        1.0134,1.0134,1.0134,1.0134,1.0134,1.0134,1.0134,1.0134,1.0201,1.0201,1.0201,1.0403,1.0470,1.0537,1.0537,1.0537,1.0604,1.0805,1.0604,1.0805,1.0805,1.0805,1.0738,1.0738,1.0671,1.0738,1.0470,0.7181,
        1.0132,1.0132,1.0132,1.0132,1.0132,1.0132,1.0132,1.0132,1.0199,1.0199,1.0199,1.0397,1.0464,1.0530,1.0530,1.0530,1.0596,1.0795,1.0596,1.0861,1.0795,1.0795,1.0728,1.0728,1.0662,1.0728,1.0464,0.7219,
        1.0131,1.0131,1.0131,1.0131,1.0131,1.0131,1.0131,1.0131,1.0196,1.0196,1.0196,1.0392,1.0458,1.0523,1.0523,1.0588,1.0588,1.0784,1.0588,1.0850,1.0784,1.0784,1.0784,1.0719,1.0654,1.0719,1.0458,0.7255,
        1.0129,1.0129,1.0129,1.0129,1.0129,1.0129,1.0194,1.0129,1.0194,1.0194,1.0194,1.0387,1.0452,1.0516,1.0516,1.0581,1.0581,1.0774,1.0581,1.0839,1.0774,1.0774,1.0774,1.0710,1.0710,1.0710,1.0452,0.7290,
        1.0127,1.0127,1.0127,1.0127,1.0127,1.0127,1.0191,1.0127,1.0191,1.0191,1.0255,1.0382,1.0446,1.0510,1.0510,1.0573,1.0573,1.0764,1.0573,1.0828,1.0764,1.0764,1.0764,1.0701,1.0701,1.0701,1.0446,0.7325,
        1.0126,1.0126,1.0126,1.0126,1.0126,1.0126,1.0189,1.0126,1.0189,1.0189,1.0252,1.0377,1.0440,1.0503,1.0503,1.0566,1.0566,1.0755,1.0566,1.0818,1.0755,1.0755,1.0755,1.0692,1.0692,1.0692,1.0440,0.7358,
        1.0124,1.0124,1.0124,1.0124,1.0124,1.0124,1.0186,1.0124,1.0186,1.0186,1.0248,1.0373,1.0435,1.0497,1.0497,1.0559,1.0559,1.0745,1.0559,1.0807,1.0745,1.0745,1.0745,1.0745,1.0683,1.0683,1.0435,0.7391,
        1.0123,1.0123,1.0123,1.0123,1.0123,1.0123,1.0184,1.0123,1.0184,1.0184,1.0245,1.0368,1.0429,1.0491,1.0491,1.0552,1.0552,1.0736,1.0552,1.0798,1.0736,1.0736,1.0736,1.0736,1.0675,1.0675,1.0491,0.7423,
        1.0121,1.0121,1.0121,1.0121,1.0121,1.0121,1.0182,1.0121,1.0182,1.0182,1.0242,1.0364,1.0424,1.0485,1.0485,1.0545,1.0545,1.0727,1.0545,1.0788,1.0727,1.0727,1.0727,1.0727,1.0667,1.0667,1.0485,0.7455,
        1.0120,1.0120,1.0120,1.0120,1.0120,1.0120,1.0180,1.0180,1.0180,1.0180,1.0240,1.0359,1.0419,1.0479,1.0479,1.0539,1.0539,1.0719,1.0539,1.0778,1.0719,1.0719,1.0719,1.0719,1.0659,1.0659,1.0479,0.7485,
        1.0118,1.0118,1.0118,1.0118,1.0118,1.0118,1.0178,1.0178,1.0178,1.0178,1.0237,1.0355,1.0414,1.0473,1.0473,1.0533,1.0533,1.0710,1.0533,1.0769,1.0710,1.0710,1.0710,1.0710,1.0651,1.0710,1.0473,0.7515,
        1.0117,1.0117,1.0117,1.0117,1.0117,1.0117,1.0175,1.0175,1.0175,1.0175,1.0234,1.0351,1.0409,1.0468,1.0468,1.0526,1.0526,1.0702,1.0526,1.0760,1.0702,1.0702,1.0702,1.0702,1.0643,1.0702,1.0468,0.7544,
        1.0116,1.0116,1.0116,1.0116,1.0116,1.0116,1.0173,1.0173,1.0173,1.0173,1.0231,1.0347,1.0405,1.0520,1.0462,1.0520,1.0520,1.0694,1.0520,1.0751,1.0694,1.0694,1.0694,1.0694,1.0636,1.0694,1.0462,0.7572,
        1.0114,1.0114,1.0114,1.0114,1.0114,1.0114,1.0171,1.0171,1.0171,1.0171,1.0229,1.0343,1.0400,1.0514,1.0514,1.0514,1.0514,1.0743,1.0514,1.0743,1.0686,1.0686,1.0686,1.0686,1.0629,1.0686,1.0457,0.7600,
        1.0113,1.0113,1.0113,1.0113,1.0113,1.0113,1.0169,1.0169,1.0169,1.0169,1.0226,1.0339,1.0395,1.0508,1.0508,1.0508,1.0508,1.0734,1.0508,1.0734,1.0678,1.0678,1.0678,1.0678,1.0621,1.0678,1.0452,0.7627,
        1.0112,1.0112,1.0112,1.0112,1.0112,1.0112,1.0168,1.0168,1.0168,1.0168,1.0223,1.0335,1.0391,1.0503,1.0503,1.0503,1.0503,1.0726,1.0503,1.0726,1.0726,1.0670,1.0670,1.0670,1.0615,1.0670,1.0447,0.7654,
        1.0110,1.0110,1.0110,1.0110,1.0110,1.0110,1.0166,1.0166,1.0166,1.0166,1.0221,1.0331,1.0387,1.0497,1.0497,1.0497,1.0497,1.0718,1.0552,1.0718,1.0718,1.0663,1.0663,1.0663,1.0608,1.0663,1.0442,0.7680,
        1.0109,1.0109,1.0109,1.0109,1.0109,1.0109,1.0164,1.0164,1.0164,1.0164,1.0219,1.0328,1.0383,1.0492,1.0492,1.0492,1.0492,1.0710,1.0546,1.0710,1.0710,1.0656,1.0656,1.0656,1.0601,1.0656,1.0437,0.7705,
        1.0108,1.0108,1.0108,1.0108,1.0108,1.0108,1.0162,1.0162,1.0162,1.0162,1.0216,1.0324,1.0378,1.0486,1.0486,1.0486,1.0486,1.0703,1.0541,1.0703,1.0703,1.0703,1.0649,1.0649,1.0595,1.0649,1.0432,0.7784,
        1.0107,1.0107,1.0107,1.0107,1.0107,1.0107,1.0160,1.0160,1.0160,1.0160,1.0214,1.0321,1.0374,1.0481,1.0481,1.0481,1.0481,1.0695,1.0535,1.0695,1.0695,1.0695,1.0642,1.0642,1.0588,1.0642,1.0428,0.7807,
        1.0106,1.0106,1.0106,1.0106,1.0106,1.0106,1.0159,1.0159,1.0159,1.0159,1.0212,1.0317,1.0423,1.0476,1.0476,1.0476,1.0476,1.0688,1.0529,1.0688,1.0688,1.0688,1.0635,1.0635,1.0582,1.0635,1.0423,0.7831,
        1.0105,1.0105,1.0105,1.0105,1.0105,1.0105,1.0157,1.0157,1.0157,1.0209,1.0209,1.0366,1.0419,1.0471,1.0471,1.0471,1.0524,1.0681,1.0524,1.0681,1.0681,1.0681,1.0628,1.0628,1.0576,1.0628,1.0419,0.7853,
        1.0104,1.0104,1.0104,1.0104,1.0104,1.0104,1.0155,1.0155,1.0155,1.0207,1.0207,1.0363,1.0415,1.0466,1.0466,1.0466,1.0518,1.0674,1.0518,1.0674,1.0674,1.0674,1.0622,1.0622,1.0570,1.0622,1.0415,0.7876,
        1.0103,1.0103,1.0103,1.0103,1.0103,1.0103,1.0154,1.0154,1.0154,1.0205,1.0205,1.0359,1.0410,1.0462,1.0462,1.0462,1.0513,1.0667,1.0513,1.0667,1.0667,1.0667,1.0615,1.0615,1.0564,1.0615,1.0410,0.7897,
        1.0102,1.0102,1.0102,1.0102,1.0102,1.0102,1.0152,1.0152,1.0152,1.0203,1.0203,1.0355,1.0406,1.0457,1.0457,1.0457,1.0508,1.0660,1.0508,1.0660,1.0660,1.0660,1.0609,1.0609,1.0558,1.0609,1.0406,0.7919,
        1.0101,1.0101,1.0101,1.0101,1.0101,1.0101,1.0151,1.0151,1.0151,1.0201,1.0201,1.0352,1.0402,1.0452,1.0452,1.0452,1.0503,1.0653,1.0503,1.0704,1.0653,1.0653,1.0603,1.0603,1.0553,1.0603,1.0402,0.7940,
        1.0100,1.0100,1.0100,1.0100,1.0100,1.0100,1.0149,1.0149,1.0149,1.0199,1.0199,1.0348,1.0398,1.0448,1.0448,1.0498,1.0498,1.0647,1.0498,1.0697,1.0647,1.0647,1.0647,1.0597,1.0547,1.0597,1.0398,0.7960]
