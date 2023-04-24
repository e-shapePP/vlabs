# REACT documentation

The REACT tool aims to rely on open-source, publicly available datasets to conduct rapid environmental flows assessment to enable timely stakeholder involvement. This is a repository containing scripts or tools used for REACT, the Deltares tool for environmental flows assessment at multiple spatial scales.

## Getting started
First, install the environment.
```
conda env create -f react_environment.yml
```

Next, ensure you have a google earth engine account. If not, sign-up [here](https://code.earthengine.google.com/register). You will need it to authenticate/initialize earth engine packages within the tool.

Finally, use ```example.py``` to run the script with your own region examples. A section of the Pineios River in Greece is used as a default example, but feel free to change it with your own defined geometries from e.g. a shapefile or GeoJSON file.

## Contact
For further enquiries regarding the use of the software, please approach the following developers: Mario Fuentes Monjaraz (Mario.FuentesMonjaraz@deltares.nl) and Robyn Gwee (Robyn.Gwee@deltares.nl). The REACT tool has been developed with support from the Deltares Water Resources Strategic Research Programme.
