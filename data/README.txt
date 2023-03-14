This folder includes: 
- m2cai16-tool training data:
  - 10 cholecystectomy videos
  - 10 tool presence annotations
- m2cai16-tool testing data:
  - 5 cholecystectomy videos
  - 5 tool presence annotations
- MATLAB scripts to perform the evaluation
- a README file

================================================
DATASET DESCRIPTION
================================================

This dataset contains 15 videos of cholecystectomy procedures from University Hospital of Strasbourg/IRCAD (Strasbourg, France). The dataset is split into two parts: training subset (containing 10 videos) and testing subset (5 videos). For the sake of fairness, please do not use any of the testing videos during your training process.

The videos are recorded at 25 fps and the tool presence annotation is generated at 1 fps. The tool annotation indicates the presence of seven tools on an image. The seven tools are: (1) grasper, (2) bipolar, (3) hook, (4) scissors, (5) clipper, (6) irrigator, and (7) specimen bag. 

The annotation file contains a table, consisting of 8 columns. Every row (except for the header of the table) contains an annotation for an image in the video. 
The first column indicates the frame index of the annotated image in the video. The frame index is defined under a 0-based system. The other columns are the binary labels for the tools (0=not present; 1=present). 

================================================
EVALUATION SCRIPT
================================================

The main script is Main.m. The script assumes that the annotation files and the result files are in the same folder. It also assumes the naming formats of the annotation and the result files are tool_video_<#ID>.txt and the tool_video_<#ID>_pred.txt, respectively. The content of the result files should be in the same format as the format in the annotation files. However, since we are computing the average precision metrics, the content of the result files should be the confidence of the tool presence. 

We have included two result file examples in the folder: (tool_video_01_pred.txt and tool_video_02_pred.txt). These examples are generated with random values.

================================================
LICENSE AND REFERENCES
================================================

This dataset could only be generated thanks to the continuous support from the surgeons from our partner hospital. In order to properly credit them for their efforts, you are kindly requested to cite the work that led to the generation of this dataset:
- A.P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. de Mathelin, N. Padoy. EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos. IEEE Trans. on Medical Imaging 2016.

The m2cai16-tool dataset is publicly released under the Creative Commons licence CC-BY-NC-SA 4.0. This implies that:
- the dataset cannot be used for commercial purposes,
- the dataset can be transformed (additional annotations, etc.),
- the dataset can be redistributed as long as it is redistributed under the same license with the obligation to cite the contributing work which led to the generation of the m2cai16-tool dataset (mentioned above).

By downloading and using this dataset, you agree on these terms and conditions.

================================================
CONTACT
================================================

The official webpage of the challenge can be found here: http://camma.u-strasbg.fr/m2cai2016
To see a list of recognition results on this dataset, please visit our web page: http://camma.u-strasbg.fr/result-list-m2cai16-tool

Any questions regarding the dataset or the challenge can be sent to: m2cai2016@gmail.com
