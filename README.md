# sen2gpp documentation

The sen2gpp tool aims to performe a Spatio-Temporal Upscaling of Flux Tower Gross Primary Productivity Measurements from the European Fluxes Database Cluster. For details of the methodology see Spinosa, A.; Fuentes-Monjaraz, M.A.; El Serafy, G. Assessing the Use of Sentinel-2 Data for Spatio-Temporal Upscaling of Flux Tower Gross Primary Productivity Measurements. Remote Sens. 2023, 15, 562. https://doi.org/10.3390/rs15030562. 

## Getting started
First, install the environment. The environment is located in the sen2gpp/Environment folder.
```
conda env create -f e-shape_environment_linux.yaml
conda activate Deltares_GPP_cmd
```

Save the input data in the sen2gpp/Input/EV and sen2gpp/Input/NEE, respectively for the environmental data and Net Ecosystem Exchange data files.  

Save the configuration file in the sen2gpp/Configs folder as Configuration_file.cfg. Run the code sen2gpp.
```
python sen2gpp.py
```
The outputs will be saved in the sen2gpp/Output folder.

For details on the implementation see Manual.doc file (https://github.com/e-shapePP/vlabs/blob/main/Manual.docx). 

## Contact
For further enquiries regarding the use of the software, please approach the following developers: Mario Fuentes Monjaraz (Mario.FuentesMonjaraz@deltares.nl) and Anna Spinosa (Anna.Spinosa@deltares.nl). The sen2gpp tool has been developed with support from the e-shape project. 
