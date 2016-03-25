# DRREP

Gene Sher, Copyright (C) 2016
This file describes how to work with DRREP v1.0
2016-03-13
------------------------------------------------
=====
DRREP v1.0   2016-03-13
The DRREP B-cell Epitope predictor using Deep Network with analytically calculated hidden synaptic weights, is written in JuliaLang (www.julialang.org)
The original program was developed and ran on a Linux system (Ubuntu 12.04) 

NOTE: Though the training and meta-parameter tuning of DRREP, due to the analytical rather than iterative methodology, is an order of magnitude faster than that of SVM, due to the architectural choices and a highly explicit XML based parameter encoding, the loading of the system has a noticeable I/O overhead. At the time of this writing, the loading overhead of DRREP is nearly a minute long on a Intel Core-2 system running at 2.0 Ghz. Substantial improvements are possible through the choice of more optimal data-preprocessing, a more concise parameter encoding, and GPU capable re-implementation. We recognize these shortcomings in this early implementation of the DRREP system.

=====
Installing JuliaLang and LightXML module.

1. Go to: http://julialang.org/downloads/platform.html
2. Choose the operating system running on your computer.
3. Download/Install Julia, following the instructions provided for that OS on the JuliaLang site.
4. Change directory to DRREP
5. Start julia
6. Execute from inside julia: Pkg.add("LightXML")
7. Exit julia by executing Ctrl-D

You now have julia installed, and the LightXML package added to your system. You can now execute the Load_DRREP.jl file.

===== 
run the DRREP

Usage: 
1. Move to DRREP_V1/ folder
2.execute: julia Load_DRREP.jl [InputFilePath] [OutputFilePath] [ThresholdOffsetMultiplier]
Input file is expected to have the protein sequence in a single row, for example: VGHDSGGFSTTVSTGQSVPDPQVGITTMKDLKANRGKMDVSGVQAPVGAITTIEDPVLAKKVPETFPELKPGESRHTSDHMSIYKFMGRSHFLCTFTFNSNNKEKEYTFPITL
An example, Custom_In.csv can be found in the Input_Files folder. Finally, the ThresholdOffsetMultiplier argument is any floating point value, and is the multiplier of the StdDev.
The default Threshold is set to the mean of the scores for the sequence DRREP is analysing, and the default ThresholdOffsetMultiplier = 0, resulting in the ThresholdOffset = 0*StdDev.
The ThresholdOffset is a value added to the MeanScore, resulting in the threshold being set to: Threshold = Threshold+ThresholdOffset.

OutputFilePath is the filepath to where the output file should be written. Output file will produce output in two formats:
1. Sequence with residue classification.
2. Ranked and scored epitopes, with index start values.

If you wish to load DRREP from the outside of DRREP_V1 folder, then use the following command: julia [Path to Load_DRREP.jl] [InputFilePath] [OutputFilePath] [ThresholdOffsetMultiplier]

=====
FEEDBACK
If you have any feedback, contact Gene Sher at gsher@knights.ucf.edu.

=====
Example use case:
>julia Load_DRREP.jl ./Input_Files/Custom_In.csv ./Custom_out.txt
>Starting
>Input URL: ./Input_Files/Custom_In.csv
>Output URL: ./Custom_Out.txt
>Loading Sub-DRREP.
>Postprocessing sliding window outputs for SubDrep-1 of 10:.Done.
>Loading Sub-DRREP.
>Postprocessing sliding window outputs for SubDrep-2 of 10:.Done.
>Loading Sub-DRREP.
>Postprocessing sliding window outputs for SubDrep-3 of 10:.Done.
>Loading Sub-DRREP.
>Postprocessing sliding window outputs for SubDrep-4 of 10:.Done.
>Loading Sub-DRREP.
>Postprocessing sliding window outputs for SubDrep-5 of 10:.Done.
>Loading Sub-DRREP.
>Postprocessing sliding window outputs for SubDrep-6 of 10:.Done.
>Loading Sub-DRREP.
>Postprocessing sliding window outputs for SubDrep-7 of 10:.Done.
>Loading Sub-DRREP.
>Postprocessing sliding window outputs for SubDrep-8 of 10:.Done.
>Loading Sub-DRREP.
>Postprocessing sliding window outputs for SubDrep-9 of 10:.Done.
>Loading Sub-DRREP.
>Postprocessing sliding window outputs for SubDrep-10 of 10:.Done.



Output file is Custom_Out.txt, with contents:
DRREP Legend:
1=amino acid position
2=Amino acid Sequence
3=Epitope prediction based on MeanScoreThreshold (0.10028097000562791) + StdDev_Multiplier*StdDev (0.031094896390824467)

1         11         21         31         41         
VGHDSGGFSTTVSTGQSVPDPQVGITTMKDLKANRGKMDVSGVQAPVGAI
.EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE.......EEEEEE..
51         61         71         81         91         
TTIEDPVLAKKVPETFPELKPGESRHTSDHMSIYKFMGRSHFLCTFTFNS
....EEEEEEEEEEEEEEEEE.............................
101         111         121         131         141         
NNKEKEYTFPITL
EEEEEEEEEEEE.

Rank  Sequence    Start position  Score
1    GHDSGGFSTTVSTGQSVPDPQVGITTMKDLKANR    2   0.3307502542908446
2    NNKEKEYTFPIT    101   0.3263443612096
3    DPVLAKKVPETFPELKP    55   0.22047887903915656
4    VQAPVG    43   0.15317254097789393
