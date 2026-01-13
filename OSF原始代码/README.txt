Dynamic switching between brain networks predicts creative ability across cultures

This code utilizes Dynamic Functional Analysis developed by Shine et al (2015) to estimate the switching frequency, which measures the transition number between segregated and integrated states. In this analysis, we specifically focus on the functional interaction between the default mode and executive control networks, so we do not use whole-brain data. Preprocessed data is provided in the data file, where the mean time series were extracted from 106 regions of interest (ROIs) associated with the two networks of interest. These ROIs were obtained in MNI standard space from 300 cortical parcels with 17 networks. Due to our limited coding expertise, you may find that the code is not well-documented. Nevertheless, we encourage you to try it out, as these codes have been tested and can run quickly. If you have any questions, please contact the first author at chenqunlin@swu.edu.cn.

A working version of MATLAB and R is required for this analysis. If you have access to MATLAB, please visit the MathWorks website and download the MATLAB installer appropriate for your operating system. Our MATLAB version is 9.0.0.341360 (R2016a). Additionally, you should install R and RStudio. You can find instructions for installation at the following link: https://rstudio-education.github.io/hopr/starting.html

Regarding the data: Since the original data is too large to be hosted here, we have provided preprocessed data suitable for MATLAB analysis. Please download these from the data file. You will find both resting-state fMRI data and task fMRI data there. Place them inside the "yourdata/" folder.

As for the code: It includes a dyfctool package containing several scripts that serve as dependencies for the analysis. Additionally, you can find scripts like rsfc_DynamicIndex and tsfc_DynamicIndex used for computing indices.

Once you have completed these preparations, you can proceed step by step with the analysis.

Part Ⅰ, an illustration of data analysis on resting-state fMRI data, involves the following five steps:
1.Download the resting-state fMR data, which were obtained in MNI standard space from 300 cortical parcels with 17 networks, to your local computer path. We provide three datasets that are publicly available, labeled with the prefix "rsfc" for resting-state data.
2.Load the dependent function package "dyfctool," or place these functions in the directory where your analysis code resides.
3.Run rsfc_DynamicIndex.m. Before execution, ensure that you load the data and specify the output file path.
4.After completion, a file named Trady_results_mtd.mat will be generated. The first column contains the quantity of state 1, the second column contains the quantity of state 2, and the third column contains the frequency of switches.
5.Consolidate these three columns of data into a behavioral data table.
6.Execute the corresponding section of code in the Statistic Analysis R Code.

Part Ⅱ, an illustration of data analysis on task-fMRI data, involves the following seven steps:

1.Download the task fMRI data file, obtained in MNI standard space, containing 300 cortical parcels with 17 networks, to your local computer path. We provide preprocessed task-based fMRI data comprising three runs. Please refer to the file task_data.mat. In this file, the first three columns represent the three runs, and the fourth column contains subject identifiers.
2.Load the dependent function package "dyfctool," or place these functions in the directory where your analysis code resides.
3.Run tsfc_DynamicIndex.m. Before execution, ensure that you load the data and specify the output file path. Unlike resting-state fMRI data, you need to run this code independently for each run.
4.After completion, a file named Trady_results_mtd.mat will be generated for each run. Each file contains columns representing the quantity of state 1, state 2, and the frequency of switches.
5.Next, calculate the frequency of switches for the "NU" condition and the "OC" condition by running dystats_task.m. In this code, ensure to load the behavioral data "Multi_AUT.mat" and the reaction time data "meanRTrunx" for each run. Finally, obtain the result file "dyresults.mat" for each run.
6.Consolidate the data from each run into a behavioral data table.
7.Execute the corresponding section of code in the Statistic Analysis R Code.