#Import Python packages used in the code
#Original packages by Matthias Cuntz
import configparser
import os.path
import numpy as np
import pandas as pd
import pyjams as pj
import hesseflux as hf

#Additional packages by Mario Alberto Fuentes Monjaraz 
import datetime as dt
import altair as alt
import math
from pyproj import Proj
import FFP_Python.calc_footprint_FFP_climatology as ffpmodule
import ee
import matplotlib.pyplot as plt

class Gpp:
    def __init__(self):
        self.version = '0.1'

    #Function to identify columns with specific beggining
    #Funtions defined here can be used in each module of the workflow
    def _findfirststart(self, starts, names):
        """
        Finds elements in a list of "names" that begin with elements in "start" 
        """
        hout = []
        for hh in starts:
            for cc in names:
                if cc.startswith(hh):
                    hout.append(cc)
                    break
        return hout
    
    def time_counting(self, t01, t02, message):

        strin = ( '{:.1f} [minutes]'.format((t02 - t01) / 60.)                                           
                if (t02 - t01) > 60.
                else '{:d} [seconds]'.format(int(t02 - t01))
            )
        
        print(f'\n     {message} ', strin, end='\n\n')

        return
    
    def read_parameters(self, configfilepath):
        # 1a)   Read configuration file
        # 1a.a) Read from command-line interpreter (It must include in the cosole "Deltares_GPP.py Configuration_file.cfg" and config file must be located in the same path)
        # if len(sys.argv) <= 1:
        #     raise IOError('Input configuration file must be given.')
        # configfile = sys.argv[1]
                                                                                             
        # 1a.b)Read from directory path
        configfile = configfilepath                                               
        
        # 1a.c)Read from gui window. Activate to manually select the congifuration file
        # configfile = hf.files_from_gui(initialdir='.', title='configuration file')
        
        config = configparser.ConfigParser(interpolation=None)                                             
        config.read(configfile)                                                                            

        # 1b) Assign variables from configuration file
 
        # Input and output paths
        self.ID            = config['GENERAL'].get('ID',        ".")
        self.inputdir      = config['GENERAL'].get('inputdir',  ".")                                            
        self.outputdir     = config['GENERAL'].get('outputdir', ".")

        # Meteorological data
        self.meteo_file    = config['METHEO'].get('meteo_file', ".")
        
        # Program switches. They activates each module of the workflow
        #------------------------------------------------------------
        self.outlier           = config['POSTSWITCH'].getboolean('outlier',              True)
        self.ustar             = config['POSTSWITCH'].getboolean('ustar',                True)
        self.ustar_non_annual  = config['POSTSWITCH'].getboolean('ustar_non_annual',     True)                  
        self.partition         = config['POSTSWITCH'].getboolean('partition',            True)                                     
        self.fill              = config['POSTSWITCH'].getboolean('fill',                 True)                                     
        self.fluxerr           = config['POSTSWITCH'].getboolean('fluxerr',              True)
        #------------------------------------------------------------
        self.daily_gpp                 =  config['POSTSWITCH'].getboolean('daily_gpp',                   True)  
        #------------------------------------------------------------
        self.climatological_footprint  =  config['POSTSWITCH'].getboolean('climatological_footprint',    True) 
        self.calculated_ffp            =  config['POSTSWITCH'].getboolean('calculated_ffp',             False)    
        #------------------------------------------------------------
        self.vegetation_indices        =  config['POSTSWITCH'].getboolean('vegetation_indices',          True)
        #------------------------------------------------------------
        self.environmental_variables_station     =  config['POSTSWITCH'].getboolean('environmental_variables_station',           True)
        self.environmental_variables_satellite   =  config['POSTSWITCH'].getboolean('environmental_variables_satellite',         True)
        self.tower_observations                  =  config['POSTSWITCH'].getboolean('tower_observations',                        True)
        self.rei_gpp_switch                      =  config['POSTSWITCH'].getboolean('rei_gpp_switch ',                           True)
        self.fal_gpp_switch                      =  config['POSTSWITCH'].getboolean('fal_gpp_switch ',                           False)
        self.las_gpp_switch                      =  config['POSTSWITCH'].getboolean('las_gpp_switch ',                           False)
        
        self.df_rainfall_station_switch = config['POSTSWITCH'].getboolean('df_rainfall_station_switch',      True)
        self.df_meteo_station_switch    = config['POSTSWITCH'].getboolean('df_meteo_station_switch',         True)
        self.df_rainfall_CHIRPS_switch  = config['POSTSWITCH'].getboolean('df_rainfall_CHIRPS_switch',       True)
        self.df_temp_MODIS_switch       = config['POSTSWITCH'].getboolean('df_temp_MODIS_switch',            True)
        self.df_meteo_tower_switch      = config['POSTSWITCH'].getboolean('df_meteo_tower_switch',           True)
        
        #------------------------------------------------------------
        self.correlation_analysis        =  config['POSTSWITCH'].getboolean('correlation_analysis',          True)
        self.correlation_analysis_simple =  config['POSTSWITCH'].getboolean('correlation_analysis',          True)
        self.calibration_validation      =  config['POSTSWITCH'].getboolean('calibration_validation',        True) 
        self.MODIS_analysis              =  config['POSTSWITCH'].getboolean('MODIS_analysis',                True) 
        #------------------------------------------------------------
        self.timeseries_thirty           =  config['POSTSWITCH'].getboolean('timeseries_thirty',        True) 
        self.timeseries_fifteen          =  config['POSTSWITCH'].getboolean('timeseries_fifteen',       True) 
        self.mapping_GPP                 =  config['POSTSWITCH'].getboolean('mapping_GPP',              True)
        self.classification_maps         =  config['POSTSWITCH'].getboolean('classification_maps',      True)
        self.maps_from_features          =  config['POSTSWITCH'].getboolean('maps_from_features',       True)
        self.mapping_GPP_thirty          =  config['POSTSWITCH'].getboolean('mapping_GPP_thirty',       True)
        self.mapping_GPP_fifteen         =  config['POSTSWITCH'].getboolean('mapping_GPP_fifteen',      True)
        self.export_maps_to_drive        =  config['POSTSWITCH'].getboolean('export_maps_to_drive',    False) 
        
        # input file format
        self.eufluxfile  = config['POSTIO'].get('eufluxfile',            '')
        self.timeformat  = config['POSTIO'].get('timeformat',  '%Y%m%d%H%M')
        self.sep         = config['POSTIO'].get('sep',                  ',')
        self.skiprows    = config['POSTIO'].get('skiprows',              '')
        self.undef       = config['POSTIO'].getfloat('undef',        -9999.)
        self.swthr       = config['POSTIO'].getfloat('swthr',           10.)
        self.outputfile  = config['POSTIO'].get('outputfile'             '')
        self.outputname  = config['POSTIO'].get('outputname'             '')
        self.outundef    = config['POSTIO'].getboolean('outundef',    False)
        self.outflagcols = config['POSTIO'].getboolean('outflagcols', False)

        # input file variables 
        self.carbonflux     = config['POSTVAR'].get('carbonflux',          'FC')                                  
                                                                                                                                          
        # remove information on a variable 
        self.remove_SW_IN   = config['POSTVAR'].getboolean('remove_SW_IN', False)                               
                                                                                                       
        # mad parameters
        self.nscan = config['POSTMAD'].getint('nscan', 15)
        self.nfill = config['POSTMAD'].getint('nfill',  1)
        self.z     = config['POSTMAD'].getfloat('z',    7)
        self.deriv = config['POSTMAD'].getint('deriv',  2)
        
        # ustar parameters
        self.ustarmin       = config['POSTUSTAR'].getfloat('ustarmin',          0.1)
        self.nboot          = config['POSTUSTAR'].getint('nboot',                 1)
        self.plateaucrit    = config['POSTUSTAR'].getfloat('plateaucrit',      0.95)
        self.seasonout      = config['POSTUSTAR'].getboolean('seasonout',     False)                                                    
        self.applyustarflag = config['POSTUSTAR'].getboolean('applyustarflag', True)

        # gap-filling parameters
        self.sw_dev  = config['POSTGAP'].getfloat('sw_dev',  50.)
        self.ta_dev  = config['POSTGAP'].getfloat('ta_dev',  2.5)
        self.vpd_dev = config['POSTGAP'].getfloat('vpd_dev', 5.0)
        self.longgap = config['POSTGAP'].getint('longgap',   60)
        
        # partitioning parameters 
        self.nogppnight = config['POSTPARTITION'].getboolean('nogppnight', False)
        
        # daily gpp computation parameters
        self.carbonfluxlimit  = config['DAILYGPP'].getint('carbonfluxlimit',         100)                       
        self.respirationlimit = config['DAILYGPP'].getint('respirationlimit',        100)                       
        self.rolling_window_gpp   = config['DAILYGPP'].getint('rolling_window_gpp',    3)
        self.rolling_center_gpp   = config['DAILYGPP'].getboolean('rolling_center_gpp', True)
        self.rolling_min_periods  = config['DAILYGPP'].getint('rolling_min_periods',    1)
        
        # climatological footprint parameters
        self.altitude                        = config['CLIMATOLOGICAL'].getfloat('altitude',                      '')                                
        self.latitude                        = config['CLIMATOLOGICAL'].getfloat('latitude',                      '')                        
        self.longitude                       = config['CLIMATOLOGICAL'].getfloat('longitude',                     '')                        
        self.canopy_height                   = config['CLIMATOLOGICAL'].getfloat('canopy_height ',                '')                           
        self.displacement_height             = config['CLIMATOLOGICAL'].getfloat('displacement_height',           '')                          
        self.roughness_lenght                = config['CLIMATOLOGICAL'].getfloat('roughness_lenght ',             '')                           
        self.instrument_height_anenometer    = config['CLIMATOLOGICAL'].getfloat('instrument_height_anenometer',  '')
        self.instrument_height_gas_analyzer  = config['CLIMATOLOGICAL'].getfloat('instrument_height_gas_analyzer','')
        
        self.domaint_var  = config['CLIMATOLOGICAL'].get('domaint_var','-2000,2000.,-2000.,2000.').split(',')
        self.nxt_var      = config['CLIMATOLOGICAL'].get('nxt_var',                        '1000').split(',')
        self.rst_var      = config['CLIMATOLOGICAL'].get('rst_var',             '20.,40.,60.,80.').split(',')
        for i in range(0, len(self.domaint_var)):
            self.domaint_var[i] = float(self.domaint_var[i])
        for i in range(0, len(self.rst_var)):
            self.rst_var[i] = float(self.rst_var[i])
        
        self.projection_site_UTM_zone        = config['CLIMATOLOGICAL'].get('projection_site_UTM_zone', '')     
        self.projection_site                 = '+proj=utm +zone=' + self.projection_site_UTM_zone + ' +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'       
        self.boundary_layer_height           = config['CLIMATOLOGICAL'].getfloat('boundary_layer_height',     '1500')
        
        # vegetation indices parameters  bands = 
        self.bands                           = config['VI'].get('bands','NDVI,EVI,EVI2,CLr,MNDVI,MNDWI,LSWI,NDII').split(',')
        for i in range(0, len(self.bands)):
            self.bands[i] = str(self.bands[i])
        
        self.max_cloud_coverage              = config['VI'].getint('max_cloud_coverage',         100)           #Default: No filter, all images available.
        self.crs                             = config['VI'].get('crs',                            '')           #EPSG:4326
        self.ndviMask                        = config['VI'].getfloat('ndviMask',                -100)           #Default: No mask
        self.mndviMask                       = config['VI'].getfloat('mndviMask',               -100)           #Default: No mask
    
        # environmental vegetation parameters
        self.rolling_window_ev_meteo         = config['EV'].getint('rolling_window_ev_meteo',            3)
        self.rolling_window_ev_meteo_sat     = config['EV'].getint('rolling_window_ev_meteo_sat',        3)
        self.rolling_window_gpp_MODIS        = config['EV'].getint('rolling_window_gpp_MODIS',           3)
        self.precipitation_data              = config['EV'].get('precipitation_data',                   '').split(',')
        self.scale_satellite_data            = config['EV'].getint('scale_satellite_data',             100)
        
        # model parameters 
        self.model_name                      = config['MODEL'].getint('model_name',                      1)
        
        # mapping parameters
        self.feature_collection              = config['MAPS'].get('feature_collection',                '')   
        self.ecosystem_extension             = config['MAPS'].getint('ecosystem_extension',          5000) 
        self.number_clusters                 = config['MAPS'].getint('number_clusters',                 4) 
        self.training_scale                  = config['MAPS'].getint('training_scale',                100)
        self.training_dataset                = config['MAPS'].getint('training_dataset',             1000)
        self.scale_getRegion                 = config['MAPS'].getint('scale_getRegion',               100)
        self.vector_scale                    = config['MAPS'].getint('vector_scale',                  100)
        
        # constant variables
        self.calculated_gpp         = False
        self.calculated_vi          = False
        self.download_maps          = True
        self.time_window_maps       = 'SM'
        self.contourlines_frequency = 0.2

        return
    

    def read_nee_file(self, inputdir, inputfile, skiprows, timeformat, sep):

        print('      Read data: ', inputfile)

        # Assert iterable                                                                                  
        if ',' in inputfile:
            inputfile     = inputfile.split(',')
            inputfile     = [ inputdir + ee.strip() for ee in inputfile]                                                                                                              
        else:                                                                                                        
            if inputfile:                                                                                   
                inputfile = [inputdir  + inputfile]
            else:
                try:
                    inputfile = pj.files_from_gui(                                                        
                        initialdir='.', title='europe-fluxdata.eu file(s)')
                except:
                    raise IOError("GUI for europe-fluxdata.eu file(s) failed.")
                
        # Identify rows in the dataframe to skipt              
        if skiprows == 'None':                                                                             
            skiprows = ''
        if skiprows:
            # to analyse int or list, tuple not working
            skiprows = json.loads(skiprows.replace('(', '[').replace(')', ']'))

        # Read input files into Panda data frame and check variable availability
        parser = lambda date: dt.datetime.strptime(date, timeformat)                               

        infile = inputfile[0]
        df = pd.read_csv(filepath_or_buffer=infile, sep=sep, skiprows=skiprows, parse_dates=[0], 
                         date_parser=parser, index_col=0, header=0) 
        
        if len(inputfile) > 1:                                                                            
            for infile in inputfile[1:]:                    
                df_aux = pd.read_csv(infile, sep=sep, skiprows=skiprows, parse_dates=[0],
                                  date_parser=parser, index_col=0, header=0)
                df = pd.concat([df,df_aux], sort=False)
                                                                                                                                                                       
        return df

    def set_constant_timestep(self, df):

        print('      Ensuring 30-minute frequncy')

        # Identify beggining and end of the time series
        df_aux = df.copy(deep=True).reset_index()
        time_ini   = df_aux.iloc[0, 0]              
        time_end   = df_aux.iloc[df.shape[0] -1,0]  
       
        # Create time series with 30-minute frequency 
        time_series = pd.date_range(time_ini, time_end, freq="30min")
        time_series = pd.DataFrame(time_series).rename(columns={0: 'TIMESTAMP_START'})
        time_series ['TIMESTAMP_END']   = time_series['TIMESTAMP_START'] + dt.timedelta(minutes = 30)
        time_series.set_index('TIMESTAMP_START',inplace=True)

        # Return'TIMESTAMP_START' as index of the origina data set and merge the dataframe with 30-minute frequency
        df_aux.set_index('TIMESTAMP_START',inplace=True)
        df_aux.drop('TIMESTAMP_END', axis =1, inplace=True)
        df_aux = pd.merge(left= time_series, right = df_aux,
                         how="left", left_index = True , right_index = True)
        
        df = df_aux.copy(deep=True)

        return df
    

    def format_nee_file(self, df, undef, swthr, remove_SW_IN):

        print('      Formating data\n')

        # The workflow works with undef values defined in the configuration file rather
        # Fill the null values (NaN) with undef values (e.g. -9999.)
        df.fillna(undef, inplace=True)
        
        # Flag.                                                                                            
        dff              = df.copy(deep=True)
        dff[:]           = 0
        dff[df == undef] = 2                                                                               
        #dff[df.isna()]   = 2

        # Dfines if it is a day / night measurement                                                                   
        hsw = ['SW_IN']                                                                                    
        hout = self._findfirststart(hsw, df.columns)                                                            
        self.isday = df[hout[0]] >= swthr

        if remove_SW_IN:                                                                                   
            df['SW_IN']=-9999.                                                                             
            df['SW_IN'].replace(-9999., np.nan, inplace=True)                                              

        # Check Ta in Kelvin
        hta = ['TA']                                                                                       
        hout = self._findfirststart(hta, df.columns)                                                                                                          
        if df[hout[0]].max() < 100.:
            self.tkelvin = 273.15
        else:
            self.tkelvin = 0.

        # Add tkelvin only where not flagged
        df.loc[dff[hout[0]] == 0, hout[0]] += self.tkelvin

        # Add vpd if not given
        hvpd = ['VPD']
        hout = self._findfirststart(hvpd, df.columns)
        if len(hout) == 0:
            hvpd = ['TA', 'RH']                                                                                                                                                         
            hout = self._findfirststart(hvpd, df.columns)
            if len(hout) != 2:
                raise ValueError('Cannot calculate VPD.')
            ta_id = hout[0]
            rh_id = hout[1]
            if df[ta_id].max() < 100.:
                tk = df[ta_id] + 273.15
            else:
                tk = df[ta_id]
            if df[rh_id].max() > 10.:
                rh = df[rh_id] / 100.
            else:
                rh = df[rh_id]
            rh_1    = 1. - rh                                                                              
            ta_1    = pj.esat(tk)
            rh_1_df = rh_1.to_frame().reset_index()
            ta_1_df = ta_1.to_frame().rename(columns={0: 'TK'})
            rh_1_df['VPD_Total'] = rh_1_df['RH'] * ta_1_df['TK']
            vpd_1_df = rh_1_df.set_index('TIMESTAMP_START')
            vpd_id = 'VPD'
            df[vpd_id] = vpd_1_df['VPD_Total']
            df[vpd_id].where((df[ta_id] != undef) | (df[rh_id] != undef),
                             other=undef, inplace=True)
            dff[vpd_id] = np.where((dff[ta_id] + dff[rh_id]) > 0, 2, 0)                                    
            df.loc[dff[vpd_id] == 0, vpd_id] /= 100.                                                       

        # Check VPD in Pa
        hvpd = ['VPD']
        hout = self._findfirststart(hvpd, df.columns)
        if df[hout[0]].max() < 10.:     # kPa
            self.vpdpa = 1000.
        elif df[hout[0]].max() < 100.:  # hPa
            self.vpdpa = 100.
        else:
            self.vpdpa = 1.                  # Pa
        df.loc[dff[hout[0]] == 0, hout[0]] *= self.vpdpa  

        # Time stepping                                                                                    
        dsec  = (df.index[1] - df.index[0]).seconds
        # Calculate the number of records per day
        self.ntday = np.rint(86400 / dsec).astype(int)  

        return df, dff
    

    def outlier_detection(self, df, dff, carbonflux, isday, undef, nscan, ntday, nfill, z, deriv):
            
        # Finds carbon flux data (e.g. NEE or FC)
        houtlier = [carbonflux]                                                                                                                  
        hout = self._findfirststart(houtlier, df.columns)
        print('      Using:', hout)

        # Applies the spike detection. Only one call to mad for all variables                          
        sflag = hf.madspikes(df[hout], flag=dff[hout], isday=isday,                                    
                                undef=undef, nscan=nscan * ntday,                                 
                                nfill=nfill * ntday, z=z, deriv=deriv, plot=False)

        for ii, hh in enumerate(hout):
            dff.loc[sflag[hh] == 2, hh]   = 3 

        return dff, sflag
    

    def frictionvelocity_filter(self, df, dff, carbonflux, isday, undef, ustarmin, nboot, plateaucrit, seasonout, applyustarflag):

        #Looking for carbonflux, u*, and temperature data
        hfilt = [carbonflux, 'USTAR', 'TA']                                                            
        hout  = self._findfirststart(hfilt, df.columns)

        assert len(hout) == 3, 'Could not find CO2 flux (NEE or FC), USTAR or TA in input file.'
        print('      Using:', hout)

        #Saves a copy of the flags of the carbonflux data
        ffsave = dff[hout[0]].to_numpy()

        #Sets a temporal flag                                                                                                   
        dff.loc[(~isday) & (df[hout[0]] < 0.), hout[0]] = 4 

        # Applies the u* filtering
        ustars, flag = hf.ustarfilter(df[hout], flag=dff[hout],                                       
                                        isday=isday, undef=undef,                                                       
                                        ustarmin=ustarmin, nboot=nboot,
                                        plateaucrit=plateaucrit,
                                        seasonout=seasonout,
                                        plot=False)

        dff[hout[0]] = ffsave                                                                          
        df  = df.assign(USTAR_TEST=flag)                                                                                                                           
        dff = dff.assign(USTAR_TEST=np.zeros(df.shape[0], dtype=int))                               
        flag = pd.DataFrame(flag).rename(columns={'USTAR': carbonflux})

        if applyustarflag:
            hustar = [carbonflux]                                                                      
            hout = self._findfirststart(hustar, df.columns)
            print('      Using:', hout)
            for ii, hh in enumerate(hout):
                dff.loc[flag[hh] == 2, hh] = 5  

        return df, dff                                                      
     

    def frictionvelocity_filter_nonannual(self, df, dff, carbonflux, undef, ustarmin, sflag, applyustarflag):

        #Looking for carbonflux, u*, and temperature data
        hfilt = [carbonflux, 'USTAR', 'TA']                                                            
        hout  = self._findfirststart(hfilt, df.columns)
        assert len(hout) == 3, 'Could not find CO2 flux (NEE or FC), USTAR or TA in input file.'
        print('      Using:', hout)

        flag = sflag.copy().multiply(0)

        #Flags when the USTAR is below ustarmin and when there is carbonflux data available for the same timestep. 
        flag.loc[(df['USTAR'] < ustarmin) & (df['USTAR'] != undef) & (df[carbonflux] != undef), carbonflux] = 2.

        df  = df.assign(USTAR_TEST=flag)               
        dff = dff.assign(USTAR_TEST=np.zeros(df.shape[0], dtype=int))

        if applyustarflag:
            hustar = [carbonflux]
            hout = self._findfirststart(hustar, df.columns)
            print('      Using:', hout)
            for ii, hh in enumerate(hout):
                dff.loc[flag[hh] == 2, hh] = 5 

        return df, dff
    

    def flux_partition(self, df, dff, carbonflux, isday, undef, nogppnight):
            
        #Looking for carbon flux, global radiation, temperature and vpd data
        hpart = [carbonflux, 'SW_IN', 'TA', 'VPD']                                                                                                                        
        hout  = self._findfirststart(hpart, df.columns)
        assert len(hout) == 4, 'Could not find CO2 flux (NEE or FC), SW_IN, TA, or VPD in input file.'
        print('      Using:', hout)

        suff = hout[0]                                                                                           

        # nighttime method
        print('      Nighttime partitioning')
        dfpartn = hf.nee2gpp(df[hout], flag=dff[hout], isday=isday,
                                undef=undef, method='reichstein',
                                nogppnight=nogppnight)

        dfpartn.rename(columns=lambda c: c + '_' + suff + '_rei', inplace=True)                        

        # falge method                                                                                 
        print('      Falge method')
        dfpartf = hf.nee2gpp(df[hout], flag=dff[hout], isday=isday,
                                undef=undef, method='falge',         
                                nogppnight=nogppnight)  

        dfpartf.rename(columns=lambda c: c + '_' + suff + '_fal', inplace=True)

        # daytime method                                                                               
        print('      Daytime partitioning')
        dfpartd = hf.nee2gpp(df[hout], flag=dff[hout], isday=isday,
                                undef=undef, method='lasslop',
                                nogppnight=nogppnight)

        dfpartd.rename(columns=lambda c: c  + '_' + suff + '_las', inplace=True) 

        df = pd.concat([df, dfpartn, dfpartf, dfpartd],  axis=1)

        # take flags from NEE or FC same flag
        for dn in ['rei', 'fal', 'las']:
            for gg in ['GPP', 'RECO']:                                                                 
                dff[gg + '_' + suff + '_'+ dn] = dff[hout[0]]                                          

        # flag GPP and RECO if they were not calculated
        for dn in ['rei', 'fal', 'las']:
            for gg in ['GPP', 'RECO']:                                                                 
                dff.loc[df['GPP' + '_' + suff + '_'+ dn] == undef, gg + '_' + suff + '_'+ dn ] = 2  

        return df, dff   


    def fill_gaps(self, df, dff, carbonflux, undef, sw_dev, ta_dev, vpd_dev, longgap):

        #Looking for meteorological data
        hfill = ['SW_IN', 'TA', 'VPD']
        hout  = self._findfirststart(hfill, df.columns)
        assert len(hout) == 3, 'Could not find SW_IN, TA or VPD in input file.'

        # if available
        rei_gpp = 'GPP_'+carbonflux+'_rei'
        rei_res = 'RECO_'+carbonflux+'_rei'
        fal_gpp = 'GPP_'+carbonflux+'_fal'
        fal_res = 'RECO_'+carbonflux+'_fal'
        las_gpp = 'GPP_'+carbonflux+'_las'
        las_res = 'RECO_'+carbonflux+'_las'

        hfill = [ carbonflux,                                                                          
                    rei_gpp,rei_res,fal_gpp,fal_res,las_gpp,las_res,
                    'SW_IN', 'TA', 'VPD']

        hout  = self._findfirststart(hfill, df.columns)
        print(f'      Using:{hout}\n')

        df_f, dff_f = hf.gapfill(df[hout], flag=dff[hout],
                                    sw_dev=sw_dev, ta_dev=ta_dev, vpd_dev=vpd_dev,
                                    longgap=longgap, undef=undef, err=False,
                                    verbose=1)

        def _add_f(c):
            return '_'.join(c.split('_')[:-3] + c.split('_')[-3:]  + ['f'])                            
        df_f.rename(columns=_add_f,  inplace=True)
        dff_f.rename(columns=_add_f, inplace=True)    

        df  = pd.concat([df,  df_f],  axis=1)
        dff = pd.concat([dff, dff_f], axis=1)

        return df, dff
    

    def estimate_error(self, df, dff, carbonflux, undef, sw_dev, ta_dev, vpd_dev, longgap):

        #Looking for meteorological data
        hfill = ['SW_IN', 'TA', 'VPD']
        hout  = self._findfirststart(hfill, df.columns)
        assert len(hout) == 3, 'Could not find SW_IN, TA or VPD in input file.'

        # if available 
        rei_gpp = 'GPP_'+carbonflux+'_rei'
        rei_res = 'RECO_'+carbonflux+'_rei'
        fal_gpp = 'GPP_'+carbonflux+'_fal'
        fal_res = 'RECO_'+carbonflux+'_fal'
        las_gpp = 'GPP_'+carbonflux+'_las'
        las_res = 'RECO_'+carbonflux+'_las'

        hfill = [ carbonflux,                                                                         
                    rei_gpp,rei_res,fal_gpp,fal_res,las_gpp,las_res,
                    'SW_IN', 'TA', 'VPD']

        hout  = self._findfirststart(hfill, df.columns)
        print('      Using:', hout)

        df_e = hf.gapfill(df[hout], flag=dff[hout],
                            sw_dev=sw_dev, ta_dev=ta_dev, vpd_dev=vpd_dev,
                            longgap=longgap, undef=undef, err=True, 
                            verbose=1)

        hdrop = ['SW_IN', 'TA', 'VPD']
        hout = self._findfirststart(hdrop, df.columns)
        df_e.drop(columns=hout, inplace=True)

        def _add_e(c):                                                                                 
            return '_'.join(c.split('_')[:-3] + c.split('_')[-3:] + ['e'])

        # rename the variables with e (error)
        colin  = list(df_e.columns)
        df_e.rename(columns=_add_e,  inplace=True)
        colout = list(df_e.columns)                                                                   
        df     = pd.concat([df, df_e], axis=1)

        # take flags of non-error columns with the same label
        for cc in range(len(colin)):
            dff[colout[cc]] = dff[colin[cc]]

        return df, dff 
    
    
    def write_gpp_files(self, df, dff, configfilepath,ID, outputdir, outputfile, outputname, tkelvin, vpdpa, undef, outundef, outflagcols):

        if not outputfile:
            try:
                outputdir = pj.directory_from_gui(initialdir='.',
                                                  title='Output directory')
            except:
                raise IOError("GUI for output directory failed.")

            outputfile = configfilepath[:configfilepath.rfind('.')]                                                
            outputfile = outputdir + '/' + ID + '_' + os.path.basename(outputfile + '.csv')                
        else:
            outputfile = outputdir +  outputfile + ID + '_' +  outputname                                  

        print('      Write output ', outputfile)

        # Back to original units
        hta = ['TA']
        hout = self._findfirststart(hta, df.columns)
        df.loc[dff[hout[0]] == 0, hout[0]] -= tkelvin
        hvpd = ['VPD']
        hout = self._findfirststart(hvpd, df.columns)
        df.loc[dff[hout[0]] == 0, hout[0]] /= vpdpa

        if outundef:
            print('      Set flags to undef.')
            for cc in df.columns:
                if cc.split('_')[-1] != 'f' and cc.split('_')[-1] != 'e':  
                    df[cc].where(dff[cc] == 0, other=undef, inplace=True)  

        if outflagcols:
            print('      Add flag columns.')

            def _add_flag(c):
                return 'flag_' + c
            dff.rename(columns=_add_flag, inplace=True)

            # no flag columns for flags
            dcol = []
            for hh in dff.columns:
                if '_TEST' in hh:                                                                                                                                       
                    dcol.append(hh)
            if dcol:
                dff.drop(columns=dcol, inplace=True)                                                      
            df = pd.concat([df, dff], axis=1)

        else:
            print('      Add flag columns for gap-filled variables.')
            occ = []
            for cc in df.columns:
                if cc.split('_')[-1] == 'f' or cc.split('_')[-1] == 'e':                                   
                    occ.append(cc)                                                                         
            dff1 = dff[occ].copy(deep=True)                                                               
            dff1.rename(columns=lambda c: 'flag_' + c, inplace=True)
            df = pd.concat([df, dff1], axis=1)

        print('      Write.')

        df.to_csv(outputfile)

        return
    
    
    def estimate_daily_gpp(self, df, carbonflux, carbonfluxlimit, respirationlimit, undef, rolling_window_gpp, rolling_center_gpp, rolling_min_periods, outputdir, ID):

        # Daily GPP and enviromental drivers
        gpp = df.copy()

        rei_gpp = 'GPP_'+carbonflux+'_rei'
        rei_res = 'RECO_'+carbonflux+'_rei'
        fal_gpp = 'GPP_'+carbonflux+'_fal'
        fal_res = 'RECO_'+carbonflux+'_fal'
        las_gpp = 'GPP_'+carbonflux+'_las'
        las_res = 'RECO_'+carbonflux+'_las'

        gpp = gpp[(gpp[carbonflux+'_f'] < carbonfluxlimit) & (gpp[carbonflux+'_f'] > -carbonfluxlimit)]                                                           
        gpp = gpp[(gpp[rei_res+'_f'] < respirationlimit) & (gpp[rei_res+'_f'] > -respirationlimit)] 

        gpp_mean = gpp[['TA_f','VPD_f','SW_IN_f']]
        gpp_sum  = gpp[[carbonflux+'_f',rei_gpp+'_f',rei_res+'_f',fal_gpp+'_f',fal_res+'_f',las_gpp+'_f',las_res+'_f']] * 12 * 30 * 60 /1000000

        gpp_mean = gpp_mean.reset_index()
        gpp_sum  = gpp_sum.reset_index()

        gpp_mean['date']  =  gpp_mean['TIMESTAMP_START'].dt.date
        gpp_sum ['date']  =  gpp_sum['TIMESTAMP_START'].dt.date

        gpp_mean.replace(undef, np.nan, inplace=True)
        gpp_sum.replace(undef, np.nan, inplace=True) 

        gpp_mean_daily = gpp_mean.drop('TIMESTAMP_START', axis=1).groupby('date').mean()
        gpp_sum_daily  = gpp_sum.drop('TIMESTAMP_START', axis=1).groupby('date').sum()

        df_gpp = pd.concat([gpp_mean_daily, gpp_sum_daily], axis=1)

        # identify beggining and end of the time series
        df_time = df_gpp.reset_index()
        time1 = df_time.iloc[0, 0]
        time2 = df_time.iloc[df_gpp.shape[0] -1,0]

        # create time series with daily frequency (Not needed if usinf gap filled variables)
        time_series = pd.date_range(time1, time2, freq="D")
        time_series = pd.DataFrame(time_series).rename(columns={0: 'date'}).set_index('date')
        df_gpp_time = pd.merge(left= time_series, right = df_gpp,
                                    how="left", left_index = True , right_index = True)

        # smoth time series  
        df_gpp_smoth  = df_gpp_time.interpolate(method='akima', order=1, limit_direction ='forward')
        df_gpp_smoth  = df_gpp_smoth.rolling(rolling_window_gpp, center=rolling_center_gpp, min_periods=rolling_min_periods).mean()

        # save file of daily GPP
        df_gpp_smoth.to_csv(outputdir + "/GPP_output/" + ID + "_GPP_daily.csv")

        # save time series plot
        model = '_rei_f'

        data_ec        = df_gpp_smoth.reset_index()
        data_co_fluxes = data_ec[['date','GPP_'+carbonflux+ model]].copy()
        data_co_fluxes = data_co_fluxes.rename(columns={
            'GPP_'+carbonflux + model: 'GPP (gC m-2 day-1)',
        })
        data_co_fluxes.head(10)

        chart = alt.Chart(data_co_fluxes).mark_bar(size=1).encode(
            x='date:T',
            y='GPP (gC m-2 day-1):Q',
            color=alt.Color(
                'GPP (gC m-2 day-1):Q', scale=alt.Scale(scheme='redyellowgreen', domain=(0, 10))),
            tooltip=[
                alt.Tooltip('date:T', title='Date'),
                alt.Tooltip('GPP (gC m-2 day-1):Q', title='GPP (gC m-2 day-1)')
            ]).properties(width=600, height=300)

        chart.save(outputdir + "/GPP_output/" + ID + "_GPP_daily.html")

        return df_gpp_smoth
    
    def read_daily_gpp_files(self, outputdir, outputfile, ID, outputname, timeformat):

        df_nee_name  = outputdir +  outputfile + ID + '_' +  outputname
        parser = lambda date: dt.datetime.strptime(date, timeformat)                               

                                                                                                                                                                     
        df = pd.read_csv(df_nee_name, parse_dates=[0], 
                         date_parser=parser, index_col=0, header=0)
        
        df_gpp_name  = outputdir + "/GPP_output/" + ID + "_GPP_daily.csv"
        parser = lambda date: dt.datetime.strptime(date, "%Y-%m-%d")                               

                                                                                                                                                                     
        df_gpp_smoth = pd.read_csv(df_gpp_name, parse_dates=[0], 
                         date_parser=parser, index_col=0, header=0)
        
        return df, df_gpp_smoth 


    def calculate_cf(
            self, 
            df, 
            dff, 
            carbonflux, 
            undef, 
            instrument_height_anenometer, 
            displacement_height, 
            roughness_lenght, 
            outputdir, 
            ID,
            boundary_layer_height,
            latitude,
            longitude,
            calculated_ffp,
            domaint_var,
            nxt_var,
            rst_var,
            projection_site
        ):

        # load carbon flux file  
        df_carbonflux = df.loc[dff[carbonflux]==0].copy(deep=True)
        df_carbonflux.replace(undef, np.nan, inplace=True)
        df_carbonflux = df_carbonflux.loc[df_carbonflux['USTAR']>0.1]
        df_carbonflux.drop(df_carbonflux.tail(1).index,inplace=True) 
        

        # metadata parameters
        fetch = 100*(instrument_height_anenometer - displacement_height) #Fetch to height ratio https://www.mdpi.com/2073-4433/10/6/299
                                                                         #https://nicholas.duke.edu/people/faculty/katul/Matlab_footprint.html  
        
        # function to add date variables to DataFrame.
        def add_date_info(df):

            df['Timestamp'] = pd.to_datetime(df['TIMESTAMP_END']) 
            df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
            df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
            df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
            df['Hour'] = pd.DatetimeIndex(df['Timestamp']).hour
            df['Minute'] = pd.DatetimeIndex(df['Timestamp']).minute

            return df
        
        # function to add ffp data to the data frame
        def add_ffp_info(df):

            df['zm']              = instrument_height_anenometer
            df['d']               = displacement_height
            df['z0']              = roughness_lenght    
            df_ec_variables       = ['Year','Month','Day','Hour','Minute',
                                     'zm','d','z0','WS','MO_LENGTH','V_SIGMA','USTAR','WD']
            ffp_variables         = ['yyyy','mm','day','HH_UTC','MM',
                                     'zm','d','z0','u_mean','L','sigma_v','u_star','wind_dir']
            df = df.loc[:,df_ec_variables].set_axis(ffp_variables , axis=1)

            return df
        
        # function to create files for the online tool 
        def online_format_by_year(df, year):
            
            # select only selected year
            df = df[df['yyyy']==year].copy(deep=True) 
            
            # compute mean velocity friction
            df_u_mean = df['u_mean'].mean()
            df['u_mean'] = df_u_mean
            
            # remove index info from file
            df_online = df.set_index('yyyy')
            
            # save file
            filepath = outputdir +'/Footprint_output/'
            df_online.to_csv(filepath + ID + "_ffp_online_data_"+ str(year) + ".csv")
            
            return df_online
        
        # function to create a climatological footprint per year
        def python_tool_by_year(df, year):
        
            df = df[df['yyyy']==year].copy(deep=True) 
            df_u_mean = df['u_mean'].mean()
            df['u_mean'] = df_u_mean
        
            # defines h (boundary layer height) in convective conditions 
            df['h_convective'] = boundary_layer_height
            
            # function that adds boundary layer height info for stable conditions
            def add_bl(df): 
                
                angular_velocity = 7.29e-05               # angular velocity of the Earth’s rotation
                radianes = (latitude * math.pi)/180
                f = 2 * angular_velocity * math.sin(radianes) # coriolis parameter
                df['h_stable'] = df['L']/3.8*(-1+( 1 + 2.28 * df['u_star']/(f*df['L']))**0.5)
                return df  
            
            # defines h (boundary layer height) in stable conditions 
            df = add_bl(df)
            
            # functions to differenciate between stable and convective conditions with the L parameter 
            def stable(L): 
                if L < 10:
                    bl = 0
                else:
                    bl = 1
                return bl
            
            def convective(L): 
                if L < 10:
                    bl = 1
                else:
                    bl = 0
                return bl
            
            df['stable']     = df['L'].apply(stable)
            df['convective'] = df['L'].apply(convective)
            
            df.replace(np.nan, -999, inplace=True) 
            
            # add h (boundary layer height) parameter
            df['h'] =  df['h_convective']*df['convective']
            df['h'] =  df['h'] + df['h_stable']*df['stable']
        
            # function to convert dataframe columns into list (used in the tool)
            def variablesToList(df):
                
                zmt      = df['zm'].to_numpy().tolist()             #0
                z0t      = df['z0'].to_numpy().tolist()             #1
                u_meant  = df['u_mean'].to_numpy().tolist()         #2
                ht       = df['h'].to_numpy().tolist()              #3
                olt      = df['L'].to_numpy().tolist()              #4
                sigmavt  = df['sigma_v'].to_numpy().tolist()        #5
                ustart   = df['u_star'].to_numpy().tolist()         #6
                wind_dirt= df['wind_dir'].to_numpy().tolist()       #7
                domaint  = domaint_var
                nxt      = nxt_var
                rst      = rst_var

                return zmt,z0t,u_meant,ht,olt,sigmavt,ustart,wind_dirt,domaint,rst
            
            ffp = variablesToList(df)
            
            # function to calcuate the footprint
            def calculateFootprint(list):
                l = list
                FFP = ffpmodule.FFP_climatology(zm=l[0], z0=None, umean = l[2], h=l[3], ol=l[4], sigmav=l[5],
                ustar=l[6], wind_dir=l[7], domain=l[8], nx=None, rs=l[9],smooth_data=1, fig=1)
                return FFP
            
            footprint = calculateFootprint(ffp)

            return footprint  
        
        # add date and labels info
        df_ffp  = add_date_info(df_carbonflux.reset_index())
        df_ffp  = add_ffp_info(df_ffp).dropna(subset=['yyyy'])
        
        # create a only file per year identified in the input files
        years = df_ffp['yyyy'].unique().tolist()
        
        for i in years:
            globals()['df_online_%s' % i] = online_format_by_year(df_ffp, i)
            print('      File: df_online_%s' % i)
        
        # create aclimatological footprint per year
        if not calculated_ffp:  
            for i in years:

                print('  \n      Footprint for the year ', i)
                globals()['df_python_footprint_%s' % i] = python_tool_by_year(df_ffp, i)

                fp = globals()['df_python_footprint_%s' % i] 

                filename = 'df_python_footprint_' + str(i)
                print('      Created file: ', filename)

                # transforme x,y values in geographical coordinates
                projection = projection_site                                                                    
                pro = Proj(projection)

                lat = latitude     
                lon = longitude    

                UTMlon, UTMlat     = pro(lon, lat)
                geo_lon, geo_lat   = pro(UTMlon, UTMlat , inverse=True)

                # tranformation per contour line 
                for n, value  in enumerate(rst_var):
                        print('      Contour line:', rst_var[n])

                        # create a data frame with the x,y data per contour line
                        x = fp['xr'][n]
                        y = fp['yr'][n]

                        d_ffp = {'x': x, 'y': y}

                        ffp = pd.DataFrame(data=d_ffp)

                        # transform x,y data into UTM
                        ffp['UTMlon'] = UTMlon
                        ffp['UTMlat'] = UTMlat

                        ffp['X'] = ffp['x'] + ffp['UTMlon'] 
                        ffp['Y'] = ffp['y'] + ffp['UTMlat']

                        # transform UTM into geographical coordinates

                        ffp['lon'], ffp['lat'] = pro(ffp['X'].values, ffp['Y'].values, inverse=True)
                        globals()['footprint_%s_%d' %(int(rst_var[n]),i)] = ffp[['lon','lat']]

                        # export file
                        ffp_export = globals()['footprint_%s_%d' %(int(rst_var[n]),i)]
                        filepath = outputdir +'/Footprint_output/'
                        ffp_export.to_csv(filepath + ID + "_footprint_" + str(int(rst_var[n])) + '_' + str(i)+ ".csv")

                        print("      Exporting: footprint_" + str(int(rst_var[n])) + '_' + str(i))

        if calculated_ffp: 
            cf_collection = {}
            for i in years:
                for n in range(len(rst_var)):
                        df_ffp_name ="Output/Footprint_output/" + ID + "_"+ 'footprint_%s_%d'%(int(rst_var[n]),i) +  ".csv"
                        # globals()['footprint_%s_%d' %(int(rst_var[n]),i)] = pd.read_csv(df_ffp_name, index_col=0, header=0)
                        cf_collection['footprint_%s_%d' %(int(rst_var[n]),i)] = pd.read_csv(df_ffp_name, index_col=0, header=0)

        return years, fetch, cf_collection 
    

    def calculate_VI(
            self,
            years, 
            fetch, 
            footprint,
            rst_var,
            longitude,
            latitude,
            max_cloud_coverage,
            ndviMask, 
            mndviMask,
            bands, 
            contourlines_frequency,
            crs,
            ID,
            outputdir
    ):

        # create climatological footprint as ee object
        for i in years:
            print('       Creating climatological footprint for %d \n'%i) 
            # create geometries 
            ## for n in range(len(rst_var)):

            ##     df = footprint['footprint_%s_%d' %(int(rst_var[n]),i)].to_numpy().tolist()
            ##     footprint['df_%s_%d_poly' %(int(rst_var[n]),i)] = ee.Geometry.Polygon(coords = df)
            ##     print('       Transforming df_%s_%d_poly' %(int(rst_var[n]),i))

            lon_lat         =  [longitude, latitude]
            point = ee.Geometry.Point(lon_lat)
            footprint['df_fetch_%s_poly' %i]   = point.buffer(fetch)
            print('       Transforming df_fetch_%s_poly' %i)

            # create areas
            ## footprint['area_%s_%d' %(int(rst_var[0]),i)] = footprint['df_%s_%d_poly' %(int(rst_var[0]),i)]
            ## print('       Creating area_%s_%d' %(int(rst_var[0]),i))


            ## for n in range(len(rst_var)-1):
            ##     footprint['area_%s_%d' %(int(rst_var[n+1]),i)]  = footprint['df_%s_%d_poly' %(int(rst_var[n+1]),i)].difference(footprint['df_%s_%d_poly' %(int(rst_var[n]),i)])
            ##     print('       Creating area_%s_%d' %(int(rst_var[n+1]),i))  

            footprint['area_100_%d' %(i)]  = footprint['df_fetch_%s_poly' %i]
            ## footprint['area_100_%d' %(i)]  = footprint['df_fetch_%s_poly' %i].difference(footprint['df_%s_%d_poly' %(int(rst_var[-1]),i)])
            print('       Creating area_100_%d ' %(i))
            print('\n') 

        # create range according to data in the input datafiles   
        start   = '%s-01-01'   %(years[0])                                              #2017-05-12 starts frequency of 10 days                                               
        end     = '%s-12-31'   %(years[-1])                                             #2017-12-18 starts frequency of 5 days
        timeSD  = [start, end]

        # create coordinates of the eddy covariance tower
        lon_lat         =  [longitude, latitude]         
        point = ee.Geometry.Point(lon_lat)

        # collections google earth engine    
        COPERNICUS_S2_L2A = 'COPERNICUS/S2_SR_HARMONIZED'        #Multi-spectral surface reflectances (https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR)
        MODIS_temp        = 'MODIS/006/MOD11A1'                  #Land surface temperature (https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD11A1)
        USAID_prec        = 'UCSB-CHG/CHIRPS/DAILY'              #InfraRed Precipitation with Station dat (https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY)
        MODIS_GPP         = 'MODIS/006/MOD17A2H'                 #Gross primary productivity(https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD17A2H)
        MODIS_NPP         = 'MODIS/006/MOD17A3HGF'               #Net primary productivity (https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD17A3HGF)

        # bands of the EO products used in the analysis
        # image.bandNames().getInfo() can be used to request bands of colections as well
        COPERNICUS_S2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'AOT', 'WVP', 'SCL', 'TCI_R', 'TCI_G', 'TCI_B', 'QA10', 'QA20', 'QA60']
        MODIS_temp_bands    = ['LST_Day_1km','QC_Day','Day_view_time','Day_view_angle','LST_Night_1km','QC_Night','Night_view_time','Night_view_angle','Emis_31','Emis_32','Clear_day_cov','Clear_night_cov']
        USAID_prec_bands    = ['precipitation']
        MODIS_GPP_bands     = ['Gpp', 'PsnNet', 'Psn_QC']
        MODIS_NPP_bands     = ['Npp', 'Npp_QC']

        # function to load data set with specified period and location
        def load_catalog(catalog, time, location, bands):
            dataset = ee.ImageCollection(catalog).filterDate(time[0],time[1]).filterBounds(location).select(bands)
            return dataset

        # cloud coverage filter function
        def cloud_filter(collection, cloud_coverage_metadata_name, threshold):
            collection_cf = collection.filterMetadata(cloud_coverage_metadata_name,'less_than', threshold)
            return collection_cf

        # function to derive VIs
        def calculateVI(image):
            '''This method calculates different vegetation indices in a image collection and adds their values as new bands'''

            # defining dictionary of bands Sentinel-2 
            dict_bands = {

                "blue"  :  'B2',                              #Blue band                        
                "green" :  'B3',                              #Green band
                "red"   :  'B4',                              #Red band
                "red1"  :  'B5',                              #Red-edge spectral band   
                "red2"  :  'B6',                              #Red-edge spectral band
                "red3"  :  'B7',                              #Red-edge spectral band    
                "NIR"   :  'B8',                              #Near-infrared band
                "NIRn"  :  'B8A',                             #Near-infrared narrow
                "WV"    :  'B9',                              #Water vapour
                "SWIR1" :  'B11',                             #Short wave infrared 1
                "SWIR2" :  'B12',                             #Short wave infrared 2
            }

            # specify bands 
            dict  = dict_bands
            blue  = dict["blue"]                              #Blue band                        
            green = dict["green"]                             #Green band
            red   = dict["red"]                               #Red band
            red1  = dict["red1"]                              #Red-edge spectral band    
            red2  = dict["red2"]                              #Red-edge spectral band
            red3  = dict["red3"]                              #Red-edge spectral band
            NIR   = dict["NIR"]                               #Near-infrared band
            NIRn  = dict["NIRn"]                              #Near-infrared band
            WV    = dict["WV"]                                #Water vapour
            SWIR1 = dict["SWIR1"]                             #Short wave infrared 1
            SWIR2 = dict["SWIR2"]                             #Short wave infrared 2

            bands_for_expressions = {

                'blue'  : image.select(blue).divide(10000),
                'green' : image.select(green).divide(10000), 
                'red'   : image.select(red).divide(10000),
                'red1'  : image.select(red1).divide(10000), 
                'red2'  : image.select(red2).divide(10000),
                'red3'  : image.select(red3).divide(10000), 
                'NIR'   : image.select(NIR).divide(10000),
                'NIRn'  : image.select(NIRn).divide(10000),
                'WV'    : image.select(WV).divide(10000),
                'SWIR1' : image.select(SWIR1).divide(10000),
                'SWIR2' : image.select(SWIR2).divide(10000)}

            # greeness related indices
            # NDVI                                                                            (Rouse et al., 1974)
            NDVI  = image.normalizedDifference([NIR, red]).rename("NDVI") 

            # EVI                                                                             
            EVI   = image.expression('2.5*(( NIR - red ) / ( NIR + 6 * red - 7.5 * blue + 1 ))', 
                    bands_for_expressions).rename("EVI")
            # EVI2                                                                            (Jiang et al., 2008)
            EVI2  = image.expression('2.5*(( NIR - red ) / ( NIR + 2.4 * red + 1 ))', 
                    bands_for_expressions).rename("EVI2")

            # greeness related indices with Sentinel-2 narrow bands / Red-edge
            # Clr
            CLr  = image.expression('(red3/red1)-1', bands_for_expressions).rename("CLr")
            # Clg
            Clg  = image.expression('(red3/green)-1', bands_for_expressions).rename("CLg")
            # MTCI
            MTCI = image.expression('(red2-red1)/(red1-red)', bands_for_expressions).rename("MTCI")
            # MNDVI                                                                          (Add reference)
            MNDVI = image.normalizedDifference([red3, red1]).rename("MNDVI")    

            # water related indices
            # MNDWI                                                                          (Add reference)
            MNDWI = image.normalizedDifference([green, SWIR1]).rename("MNDWI")    
            # NDWI OR LSWI or NDII or NDMI                                                    (Add reference)
            LSWI  = image.normalizedDifference([NIR, SWIR1]).rename("LSWI")
            # NDII                                                                            (Hunt & Qu, 2013)
            NDII   = image.normalizedDifference([NIR, SWIR2]).rename("NDII")

            image1 = image.addBands(NDVI).addBands(EVI).addBands(EVI2)
            image2 = image1.addBands(CLr).addBands(Clg).addBands(MTCI).addBands(MNDVI)
            image3 = image2.addBands(MNDWI).addBands(LSWI).addBands(NDII)

            return image3  
        
        def local_cloud_filter(s2, aoi, LOCAL_CLOUD_THRESH):

            # Describe functions
            # Function to scale the reflectance bands
            def apply_scale_factors_s2(image):
                optical_bands = image.select(['B.']).divide(10000)
                thermal_bands = image.select(['B.*']).divide(10000)
                return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

            # Function to create mask with cirrus clouds and cirrus pixels
            def extract_bit_s2_10_11(image):
                bit_position_clouds = 10
                bit_position_cirrus = 11

                # Bits 10 and 11 are clouds and cirrus, respectively.
                cloud_bit_mask = 1 << bit_position_clouds
                cirrus_bit_mask = 1 << bit_position_cirrus

                mask_clouds = image.bitwiseAnd(cloud_bit_mask).rightShift(bit_position_clouds)
                mask_cirrus = image.bitwiseAnd(cirrus_bit_mask).rightShift(bit_position_cirrus)
                mask = mask_clouds.add(mask_cirrus)
                return mask

            # Function to mask pixels with high reflectance in the blue (B2) band. The function creates a QA band
            def b2_mask(image):
                B2Threshold = 0.2
                B2Mask = image.select('B2').gt(B2Threshold)
                return image.addBands(B2Mask.rename('B2Mask'))

            # Function to create a band with ones
            def make_ones(image):
                # Create a band with ones
                ones_band = image.select('B2').divide(image.select('B2'))
                return image.addBands(ones_band.rename('Ones'))

            # Function to calculate area
            def get_area(img):
                cloud_area = make_ones(img).select('Ones').multiply(ee.Image.pixelArea()) \
                    .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=30).values().get(0)
                return img.set('area_image', ee.Number(cloud_area))

            # Function to get local cloud percentage with QA band
            def get_local_cloud_percentage(img):
                cloud_area = extract_bit_s2_10_11(img.select('QA60')).multiply(ee.Image.pixelArea()) \
                    .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=60).values().get(0)
                return img.set('local_cloud_percentage', ee.Number(cloud_area).divide(aoi.area()).multiply(100).round())

            # Function to get local cloud percentage with QA and area of image band
            def get_local_cloud_percentage_area_image(img):
                area_image = img.get('area_image')
                cloud_area = extract_bit_s2_10_11(img.select('QA60')).multiply(ee.Image.pixelArea()) \
                    .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=60).values().get(0)
                return img.set('local_cloud_percentage_ai', ee.Number(cloud_area).divide(ee.Number(area_image)).multiply(100).round())

            # Function to get local cloud percentage with B2 and area of image band
            def get_local_cloud_percentage_area_image_b2(img):
                area_image = img.get('area_image')
                cloud_area = b2_mask(img).select('B2Mask').multiply(ee.Image.pixelArea()) \
                    .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=60).values().get(0)
                return img.set('local_cloud_percentage_ai_b2', ee.Number(cloud_area).divide(ee.Number(area_image)).multiply(100).round())

            def add_ndvi(image):
                # Calculate NDVI
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                return image.addBands(ndvi)

            s2 = s2.filterBounds(aoi).map(lambda image: image.clip(aoi)).map(apply_scale_factors_s2).map(add_ndvi)
            
            # Processing
            # Mask with band 2
            extractedBitB2 = s2.select('B2').map(b2_mask)
            # Mask with QA60 band
            extractedBit = s2.select('QA60').map(extract_bit_s2_10_11)
            # Band with ones
            extractedBitones = s2.map(make_ones)
            # Calculate area
            s2 = s2.map(get_area)
            # Calculate local cloud percentage with QA band
            s2 = s2.map(get_local_cloud_percentage)
            # Calculate local cloud percentage with QA band and area image band
            s2 = s2.map(get_local_cloud_percentage_area_image)
            # Calculate local cloud percentage with B2 band and area image band
            s2 = s2.map(get_local_cloud_percentage_area_image_b2)
            # Filter images
            # LOCAL_CLOUD_THRESH = 30
            s2_filtered = s2.filter(ee.Filter.lte('local_cloud_percentage_ai', LOCAL_CLOUD_THRESH))
            s2_filtered = s2_filtered.filter(ee.Filter.lte('local_cloud_percentage_ai_b2', LOCAL_CLOUD_THRESH))

            # Show messages
            print('The original size of the collection is', s2.size().getInfo())
            # print(s2.first().getInfo())
            print('The filtered size of the collection is', s2_filtered.size().getInfo())
            print('\n')
            
            return s2_filtered 

        # function for masking non-vegetation areas
        def maskS2nonvegetation(image):

                qa    = image.select('QA60')
                scl   = image.select('SCL')
                ndvi  = image.select('NDVI')
                mndvi = image.select('MNDVI')

                cloudBitMask = 1 << 10
                cirrusBitMask = 1 << 11

                #vegetationMask1 = 4 # vegetation
                #vegetationMask2 = 5 # non-vegetated
                #vegetationMask3 = 6 # water
                #vegetationMask4 = 7 # unclassified
                #vegetationMask5 = 11 # snow

                # this mask selects vegetation + non-vegetated + water + unclassified + areas with VIs (NDVI and MNDVI) greater that a threshold set in the configuration file
                #mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).And(qa.bitwiseAnd(cloudBitMask).eq(0)).And(qa.bitwiseAnd(cirrusBitMask).eq(0)).And(ndvi.gte(ndviMask)).And(mndvi.gte(mndviMask))
                mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11)).And(qa.bitwiseAnd(cloudBitMask).eq(0)).And(qa.bitwiseAnd(cirrusBitMask).eq(0)).And(ndvi.gte(ndviMask)).And(mndvi.gte(mndviMask))
                #mask = scl.gte(4).And(qa.bitwiseAnd(cloudBitMask).eq(0)).And(qa.bitwiseAnd(cirrusBitMask).eq(0)).And(ndvi.gte(ndviMask)).And(mndvi.gte(mndviMask))
                #mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).And(mndvi.gte(0.05)) 

                vegetation = image.updateMask(mask)

                return vegetation

        # function to transform ee objects to dataframes pandas objects
        # function that transforms arrays into dataframes
        def ee_array_to_df(imagecollection, geometry, scale):

            """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""

            # select bands from the image collection
            filtered = imagecollection.select(bands)

            # function that produce functions to reduce a region with atatistics (mean, max, min, etc.)
            def create_reduce_region_function(geometry,
                                                reducer=ee.Reducer.mean(),
                                                scale=1000,
                                                crs=crs,
                                                bestEffort=True,
                                                maxPixels=1e13,
                                                tileScale=4):

                    def reduce_region_function(img):

                        stat = img.reduceRegion(
                            reducer=reducer,
                            geometry=geometry,
                            scale=scale,
                            crs=crs,
                            bestEffort=bestEffort,
                            maxPixels=maxPixels,
                            tileScale=tileScale)

                        return ee.Feature(geometry, stat).set({'millis': img.date().millis()})
                    return reduce_region_function

            # function to transfer feature properties to a dictionary.
            def fc_to_dict(fc):
                    prop_names = fc.first().propertyNames()
                    prop_lists = fc.reduceColumns(reducer=ee.Reducer.toList().repeat(prop_names.size()),selectors=prop_names).get('list')

                    return ee.Dictionary.fromLists(prop_names, prop_lists)

            # creating reduction function (reduce_VI is a function)
            reduce_VI = create_reduce_region_function(
                geometry= geometry, reducer=ee.Reducer.mean(), scale=10, crs= crs)

            # transform image collection into feature collection (tables)
            VI_stat_fc = ee.FeatureCollection(imagecollection.map(reduce_VI)).filter(
                ee.Filter.notNull(imagecollection.first().bandNames()))

            # transform feature collection into dictionary object
            VI_dict = fc_to_dict(VI_stat_fc).getInfo()

            #print(type(VI_dict), '\n')

            #for prop in VI_dict.keys():
            #    print(prop + ':', VI_dict[prop][0:3] + ['...'])

            # transform dictionary into dataframe
            VI_df = pd.DataFrame(VI_dict)

            # convert column in datatime type object
            #VI_df['datetime'] = pd.to_datetime(VI_df['time'], unit='ms')
            VI_df['date']     = pd.to_datetime(VI_df['millis'], unit='ms').dt.date

            # generate a list with the names of each band of the collection 
            list_of_bands = filtered.first().bandNames().getInfo()

            # remove rows without data inside.
            VI_df = VI_df[['date', *list_of_bands]].dropna()

            # convert the data to numeric values.
            for band in list_of_bands:
                VI_df[band] = pd.to_numeric(VI_df[band], errors='coerce', downcast ='float')

            # convert the time field into a datetime.
            #VI_df['datetime'] = pd.to_datetime(VI_df['time'], unit='ms')
            #VI_df['date']     = pd.to_datetime(VI_df['time'], unit='ms').dt.date

            # keep the columns of interest.
            #VI_df = VI_df[['datetime','date',  *list_of_bands]]

            # flag to identify if in the reduction there were pixels, or they were masked-removed
            VI_df['flag'] = 100

            # reduction in case there are two pixels from different images for the same day
            VI_df = VI_df.groupby('date').mean().reset_index().set_index('date').copy()

            return VI_df

        # applying functions 

        # request of catalogues 
        S2     = load_catalog(COPERNICUS_S2_L2A, timeSD, point, COPERNICUS_S2_bands)
        temp   = load_catalog(MODIS_temp,        timeSD, point, MODIS_temp_bands)
        prec   = load_catalog(USAID_prec,        timeSD, point, USAID_prec_bands)
        gpp_MODIS    = load_catalog(MODIS_GPP,         timeSD, point, MODIS_GPP_bands)
        npp_MODIS    = load_catalog(MODIS_NPP,         timeSD, point,  MODIS_NPP_bands)

        # calculation of vegetation indices for the collection
        S2_VI = S2.map(calculateVI)

        # # filter cloud coverage
        # cloud_coverage_metadata_name = 'CLOUDY_PIXEL_PERCENTAGE'                     # name of metadata property indicating cloud coverage in %

        # # applying cloud filter 
        # S2_VI = cloud_filter(S2_VI, cloud_coverage_metadata_name, max_cloud_coverage)   # max cloud coverage defined in the Config file

        # apply cloud local filter
        S2_VI = local_cloud_filter(S2_VI, footprint['area_100_%d' %(i)], max_cloud_coverage)

        # applying mask 
        S2_VI = S2_VI.map(maskS2nonvegetation)

        # select bands for the analysis of interdependency and regression models 
        #bands = ['NDVI','EVI','EVI2','CLr','CLg','MTCI','MNDVI','MNDWI','LSWI','NDII']
        bands = ['NDVI','EVI','EVI2','CLr','MNDVI','MNDWI','LSWI','NDII', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

        # applying funtion ee_array_to_df for each climatological footprint
        for i in years:
            ## for n in range(len(rst_var)):

            ##         area = footprint['area_%s_%d' %(int(rst_var[n]),i)]
            ##         footprint['S2_VI_df_area_%s_%d' %(int(rst_var[n]),i)]  = ee_array_to_df(S2_VI, area, 10)          
            ##         print('       Retrieving info from area_%s_%d into dataframe:' %(int(rst_var[n]),i))
            ##         print('       S2_VI_df_area_%s_%d' %(int(rst_var[n]),i))


            area = footprint['area_100_%d' %i]
            footprint['S2_VI_df_area_100_%d' %i]  = ee_array_to_df(S2_VI, area, 10) 
            print('       Retrieving info from area_100_%d into dataframe:' %i)
            print('       S2_VI_df_area_100_%d' %i)
            print('\n')

        # loop to derive a weighted value of each VI per climatological footprint
        for i in years:

            # create an empty file
            footprint['dfvi_%s'%i] = pd.DataFrame(columns=bands)

            ## for n in range(len(rst_var)):

            ##         df = footprint['S2_VI_df_area_%s_%d' %(int(rst_var[n]),i)].multiply(contourlines_frequency)  
            ##         print('       Concat S2_VI_df_area_%s_%d' %(int(rst_var[n]),i))
            ##         footprint['dfvi_%s'%i] = pd.concat([footprint['dfvi_%s'%i], df])

            ## df = footprint['S2_VI_df_area_100_%d' %i].multiply(contourlines_frequency)
            df = footprint['S2_VI_df_area_100_%d' %i]
            print('       Concat S2_VI_df_area_100_%d' %i)
            footprint['dfvi_%s'%i] = pd.concat([footprint['dfvi_%s'%i] , df])

            footprint['S2_VI_ffp_%s'%i] = footprint['dfvi_%s'%i].groupby(footprint['dfvi_%s'%i].index).sum().rename_axis('date')
            print('       Creating S2_VI_ffp_%s'%i)
            print('\n') 


        # loop to join each S2_VI_df in a single dataframe per year named as df_VI
        df_VI = pd.DataFrame(columns=bands)

        ndvi_threshold = -100                                                            # Threshold applied in the time series. Not used in the analysis

        for i in years:

            footprint['S2_VI_ffp_%s_join'%i] = footprint['S2_VI_ffp_%s'%i][footprint['S2_VI_ffp_%s'%i]['NDVI']>ndvi_threshold]

            def add_date_info(df):

                df = df.reset_index()
                df['date'] = pd.to_datetime(df['date'])
                df['Year'] = pd.DatetimeIndex(df['date']).year

                def timestamp(df):
                    df = df.timestamp()
                    return df

                df['timestamp'] = df['date'].apply(timestamp)
                df = df.set_index('date')

                return df

            footprint['S2_VI_ffp_%s_join'%i]  = add_date_info(footprint['S2_VI_ffp_%s_join'%i] )
            footprint['S2_VI_ffp_%s_join'%i]  = footprint['S2_VI_ffp_%s_join'%i][footprint['S2_VI_ffp_%s_join'%i]['Year'] == i]

            df = footprint['S2_VI_ffp_%s_join'%i]
            print('       Concat S2_VI_ffp_%s_join'%i)

            df_VI = pd.concat([df_VI, df]).rename_axis('date')
        print('\n') 
        print('       Creating df_VI')
        print('\n') 

        # filtering VI_eshape by flag. Removes dates when there were areas whitin the climatological footprint that were totally masked 
        df_VI_filtered = df_VI.copy()
        df_VI_filtered = df_VI_filtered[df_VI_filtered['flag']>80].drop(['flag'], axis = 1)

        # create time series with daily frequency
        time_series = pd.date_range(start=start, end=end, freq="D")
        time_series = pd.DataFrame(time_series).rename(columns={0: 'date'}).set_index('date')

        # allows to have a time series with daily frequency with gaps when the VI were not calculated or there were not S2 images
        df_VI_time = pd.merge(left= time_series, right = df_VI_filtered,
                                        how="left", left_index = True , right_index = True)  

        # interpolate values
        df_VI_interpolate = df_VI_time.interpolate(method='akima', order=1, limit_direction ='forward')
        #df_VI_interpolate_limits = df_VI_interpolate.apply(lambda x: x.interpolate(method="spline", order=6))
        #df_VI_interpolate = df_VI_interpolate.fillna(method='backfill')
        #df_VI_interpolate = df_VI_interpolate.fillna(method='ffill')

        # file to extrapolate
        df_VI_export = df_VI_interpolate.dropna().drop(['Year','timestamp'], axis = 1) 
        df_VI_export.to_csv(outputdir + '/VI_output/' + ID + "_Vegetation_indices.csv")   

        print("       Exporting: Vegetation_indices.csv")
        print('\n') 

        #---------------------------------------------------------------------------------------------
        # Save plots of VI
        def plot_timeseries_vi_multiple(df):
            # subplots.
            fig, ax = plt.subplots(figsize=(14, 6)) #Indicates the size of the plot

            if 'NDVI' in bands:
                # add scatter plots //Adds the scatter points
                ax.plot(df['NDVI'],
                            c='#00FF00', alpha=1, label='NDVI', lw=2, linestyle = ':')
            if 'EVI' in bands:
                ax.plot(df['EVI'],
                            c='red', alpha=1, label='EVI', lw=2, linestyle = ':')
            if 'EVI2' in bands:
                ax.plot(df['EVI2'],
                            c='yellow', alpha=1, label='EVI-2', lw=2, linestyle = ':')
            if 'CLr' in bands:
                ax.plot(df['CLr'],
                            c='green', alpha=1, label='CLr', lw=2, linestyle = ':')
            if 'MNDVI' in bands:
                ax.plot(df['MNDVI'],
                            c='black', alpha=1, label='MNDVI', lw=2, linestyle = ':')
            if 'MNDWI' in bands:
                ax.plot(df['MNDWI'],
                            c='#00FFFF', alpha=0.5, label='MNDWI', lw=2, linestyle = '-.')
            if 'LSWI' in bands:
                ax.plot(df['LSWI'],
                            c='blue', alpha=0.8, label='LSWI', lw=2, linestyle = '-.')
            if 'NDII' in bands:
                ax.plot(df['NDII'],
                            c='#00008B', alpha=0.8, label='NDII', lw=2, linestyle = '-.') #marker="x", markersize=2)

            ax.set_title('Vegetation Indices', fontsize=16)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('Vegetation Index', fontsize=14)
            ax.set_ylim(-1, 1)
            ax.grid(lw=0.5)
            ax.legend(fontsize=14, loc='lower right')

            # shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, .8))
            #plt.savefig(outputdir + '/VI_output/' + ID + '_VI_timeseries.png', dpi=300, format='png', bbox_inches='tight',pad_inches=0.0001)
            plt.savefig(outputdir + '/VI_output/' + ID + '_VI_timeseries.png', dpi=300)

            return plt #.show()

        plot_timeseries_vi_multiple(df_VI_export)
        return df_VI_export
    

    def calculate_VI_with_area(
            self,
            df, 
            dff, 
            carbonflux, 
            undef, 
            aoi, 
            longitude,
            latitude,
            max_cloud_coverage,
            ndviMask, 
            mndviMask,
            bands, 
            crs,
            ID,
            outputdir
    ):
        
        # load carbon flux file  
        df_carbonflux = df.loc[dff[carbonflux]==0].copy(deep=True)
        df_carbonflux.replace(undef, np.nan, inplace=True)
        df_carbonflux = df_carbonflux.loc[df_carbonflux['USTAR']>0.1]
        df_carbonflux.drop(df_carbonflux.tail(1).index,inplace=True) 
        
        # function to add date variables to DataFrame.
        def add_date_info(df):

            df['Timestamp'] = pd.to_datetime(df['TIMESTAMP_END']) 
            df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
            df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
            df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
            df['Hour'] = pd.DatetimeIndex(df['Timestamp']).hour
            df['Minute'] = pd.DatetimeIndex(df['Timestamp']).minute

            return df
        
        # add date and labels info
        df_ffp  = add_date_info(df_carbonflux.reset_index())
        df_ffp  = df_ffp.dropna(subset=['Year'])
        
        # create a only file per year identified in the input files
        years = df_ffp['Year'].unique().tolist()

        # # create aoi
        # lon_lat         =  [longitude, latitude]
        # point = ee.Geometry.Point(lon_lat)
        # aoi  = point.buffer(fetch)

        # create range according to data in the input datafiles   
        start   = '%s-01-01'   %(years[0])                                              #2017-05-12 starts frequency of 10 days                                               
        end     = '%s-12-31'   %(years[-1])                                             #2017-12-18 starts frequency of 5 days
        timeSD  = [start, end]

        # create coordinates of the eddy covariance tower
        lon_lat         =  [longitude, latitude]         
        point = ee.Geometry.Point(lon_lat)

        # collections google earth engine    
        COPERNICUS_S2_L2A = 'COPERNICUS/S2_SR_HARMONIZED'        #Multi-spectral surface reflectances (https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR)
        MODIS_temp        = 'MODIS/006/MOD11A1'                  #Land surface temperature (https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD11A1)
        USAID_prec        = 'UCSB-CHG/CHIRPS/DAILY'              #InfraRed Precipitation with Station dat (https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY)
        MODIS_GPP         = 'MODIS/006/MOD17A2H'                 #Gross primary productivity(https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD17A2H)
        MODIS_NPP         = 'MODIS/006/MOD17A3HGF'               #Net primary productivity (https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD17A3HGF)

        # bands of the EO products used in the analysis
        # image.bandNames().getInfo() can be used to request bands of colections as well
        COPERNICUS_S2_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'AOT', 'WVP', 'SCL', 'TCI_R', 'TCI_G', 'TCI_B', 'QA10', 'QA20', 'QA60']
        MODIS_temp_bands    = ['LST_Day_1km','QC_Day','Day_view_time','Day_view_angle','LST_Night_1km','QC_Night','Night_view_time','Night_view_angle','Emis_31','Emis_32','Clear_day_cov','Clear_night_cov']
        USAID_prec_bands    = ['precipitation']
        MODIS_GPP_bands     = ['Gpp', 'PsnNet', 'Psn_QC']
        MODIS_NPP_bands     = ['Npp', 'Npp_QC']

        # function to load data set with specified period and location
        def load_catalog(catalog, time, location, bands):
            dataset = ee.ImageCollection(catalog).filterDate(time[0],time[1]).filterBounds(location).select(bands)
            return dataset

        # cloud coverage filter function
        def cloud_filter(collection, cloud_coverage_metadata_name, threshold):
            collection_cf = collection.filterMetadata(cloud_coverage_metadata_name,'less_than', threshold)
            # Show messages
            print('The maximun cloud coverage in the image is:', max_cloud_coverage)
            print('The original size of the collection is', collection.size().getInfo())
            # print(s2.first().getInfo())
            print('The filtered size of the collection is', collection_cf.size().getInfo())
            print('\n')
            return collection_cf

        # function to derive VIs
        def calculateVI(image):
            '''This method calculates different vegetation indices in a image collection and adds their values as new bands'''

            # defining dictionary of bands Sentinel-2 
            dict_bands = {

                "blue"  :  'B2',                              #Blue band                        
                "green" :  'B3',                              #Green band
                "red"   :  'B4',                              #Red band
                "red1"  :  'B5',                              #Red-edge spectral band   
                "red2"  :  'B6',                              #Red-edge spectral band
                "red3"  :  'B7',                              #Red-edge spectral band    
                "NIR"   :  'B8',                              #Near-infrared band
                "NIRn"  :  'B8A',                             #Near-infrared narrow
                "WV"    :  'B9',                              #Water vapour
                "SWIR1" :  'B11',                             #Short wave infrared 1
                "SWIR2" :  'B12',                             #Short wave infrared 2
            }

            # specify bands 
            dict  = dict_bands
            blue  = dict["blue"]                              #Blue band                        
            green = dict["green"]                             #Green band
            red   = dict["red"]                               #Red band
            red1  = dict["red1"]                              #Red-edge spectral band    
            red2  = dict["red2"]                              #Red-edge spectral band
            red3  = dict["red3"]                              #Red-edge spectral band
            NIR   = dict["NIR"]                               #Near-infrared band
            NIRn  = dict["NIRn"]                              #Near-infrared band
            WV    = dict["WV"]                                #Water vapour
            SWIR1 = dict["SWIR1"]                             #Short wave infrared 1
            SWIR2 = dict["SWIR2"]                             #Short wave infrared 2

            bands_for_expressions = {

                'blue'  : image.select(blue).divide(10000),
                'green' : image.select(green).divide(10000), 
                'red'   : image.select(red).divide(10000),
                'red1'  : image.select(red1).divide(10000), 
                'red2'  : image.select(red2).divide(10000),
                'red3'  : image.select(red3).divide(10000), 
                'NIR'   : image.select(NIR).divide(10000),
                'NIRn'  : image.select(NIRn).divide(10000),
                'WV'    : image.select(WV).divide(10000),
                'SWIR1' : image.select(SWIR1).divide(10000),
                'SWIR2' : image.select(SWIR2).divide(10000)}

            # greeness related indices
            # NDVI                                                                            (Rouse et al., 1974)
            NDVI  = image.normalizedDifference([NIR, red]).rename("NDVI") 

            # EVI                                                                             
            EVI   = image.expression('2.5*(( NIR - red ) / ( NIR + 6 * red - 7.5 * blue + 1 ))', 
                    bands_for_expressions).rename("EVI")
            # EVI2                                                                            (Jiang et al., 2008)
            EVI2  = image.expression('2.5*(( NIR - red ) / ( NIR + 2.4 * red + 1 ))', 
                    bands_for_expressions).rename("EVI2")

            # greeness related indices with Sentinel-2 narrow bands / Red-edge
            # Clr
            CLr  = image.expression('(red3/red1)-1', bands_for_expressions).rename("CLr")
            # Clg
            Clg  = image.expression('(red3/green)-1', bands_for_expressions).rename("CLg")
            # MTCI
            MTCI = image.expression('(red2-red1)/(red1-red)', bands_for_expressions).rename("MTCI")
            # MNDVI                                                                          (Add reference)
            MNDVI = image.normalizedDifference([red3, red1]).rename("MNDVI")    

            # water related indices
            # MNDWI                                                                          (Add reference)
            MNDWI = image.normalizedDifference([green, SWIR1]).rename("MNDWI")    
            # NDWI OR LSWI or NDII or NDMI                                                    (Add reference)
            LSWI  = image.normalizedDifference([NIR, SWIR1]).rename("LSWI")
            # NDII                                                                            (Hunt & Qu, 2013)
            NDII   = image.normalizedDifference([NIR, SWIR2]).rename("NDII")

            image1 = image.addBands(NDVI).addBands(EVI).addBands(EVI2)
            image2 = image1.addBands(CLr).addBands(Clg).addBands(MTCI).addBands(MNDVI)
            image3 = image2.addBands(MNDWI).addBands(LSWI).addBands(NDII)

            return image3  
        
        def local_cloud_filter(s2, aoi, LOCAL_CLOUD_THRESH):

            # Describe functions
            # Function to scale the reflectance bands
            def apply_scale_factors_s2(image):
                optical_bands = image.select(['B.']).divide(10000)
                thermal_bands = image.select(['B.*']).divide(10000)
                return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

            # Function to create mask with cirrus clouds and cirrus pixels
            def extract_bit_s2_10_11(image):
                bit_position_clouds = 10
                bit_position_cirrus = 11

                # Bits 10 and 11 are clouds and cirrus, respectively.
                cloud_bit_mask = 1 << bit_position_clouds
                cirrus_bit_mask = 1 << bit_position_cirrus

                mask_clouds = image.bitwiseAnd(cloud_bit_mask).rightShift(bit_position_clouds)
                mask_cirrus = image.bitwiseAnd(cirrus_bit_mask).rightShift(bit_position_cirrus)
                mask = mask_clouds.add(mask_cirrus)
                return mask

            # Function to mask pixels with high reflectance in the blue (B2) band. The function creates a QA band
            def b2_mask(image):
                B2Threshold = 0.2
                B2Mask = image.select('B2').gt(B2Threshold)
                return image.addBands(B2Mask.rename('B2Mask'))

            # Function to create a band with ones
            def make_ones(image):
                # Create a band with ones
                ones_band = image.select('B2').divide(image.select('B2'))
                return image.addBands(ones_band.rename('Ones'))

            # Function to calculate area
            def get_area(img):
                cloud_area = make_ones(img).select('Ones').multiply(ee.Image.pixelArea()) \
                    .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=30).values().get(0)
                return img.set('area_image', ee.Number(cloud_area))

            # Function to get local cloud percentage with QA band
            def get_local_cloud_percentage(img):
                cloud_area = extract_bit_s2_10_11(img.select('QA60')).multiply(ee.Image.pixelArea()) \
                    .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=60).values().get(0)
                return img.set('local_cloud_percentage', ee.Number(cloud_area).divide(aoi.area()).multiply(100).round())

            # Function to get local cloud percentage with QA and area of image band
            def get_local_cloud_percentage_area_image(img):
                area_image = img.get('area_image')
                cloud_area = extract_bit_s2_10_11(img.select('QA60')).multiply(ee.Image.pixelArea()) \
                    .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=60).values().get(0)
                return img.set('local_cloud_percentage_ai', ee.Number(cloud_area).divide(ee.Number(area_image)).multiply(100).round())

            # Function to get local cloud percentage with B2 and area of image band
            def get_local_cloud_percentage_area_image_b2(img):
                area_image = img.get('area_image')
                cloud_area = b2_mask(img).select('B2Mask').multiply(ee.Image.pixelArea()) \
                    .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=60).values().get(0)
                return img.set('local_cloud_percentage_ai_b2', ee.Number(cloud_area).divide(ee.Number(area_image)).multiply(100).round())

            def add_ndvi(image):
                # Calculate NDVI
                ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
                return image.addBands(ndvi)

            s2 = s2.filterBounds(aoi).map(lambda image: image.clip(aoi)).map(apply_scale_factors_s2).map(add_ndvi)
            
            # Processing
            # Mask with band 2
            extractedBitB2 = s2.select('B2').map(b2_mask)
            # Mask with QA60 band
            extractedBit = s2.select('QA60').map(extract_bit_s2_10_11)
            # Band with ones
            extractedBitones = s2.map(make_ones)
            # Calculate area
            s2 = s2.map(get_area)
            # Calculate local cloud percentage with QA band
            s2 = s2.map(get_local_cloud_percentage)
            # Calculate local cloud percentage with QA band and area image band
            s2 = s2.map(get_local_cloud_percentage_area_image)
            # Calculate local cloud percentage with B2 band and area image band
            s2 = s2.map(get_local_cloud_percentage_area_image_b2)
            # Filter images
            # LOCAL_CLOUD_THRESH = 30
            s2_filtered = s2.filter(ee.Filter.lte('local_cloud_percentage_ai', LOCAL_CLOUD_THRESH))
            s2_filtered = s2_filtered.filter(ee.Filter.lte('local_cloud_percentage_ai_b2', LOCAL_CLOUD_THRESH))

            # Show messages
            print('The maximun cloud coverage in the area is:', max_cloud_coverage)
            print('The original size of the collection is', s2.size().getInfo())
            # print(s2.first().getInfo())
            print('The filtered size of the collection is', s2_filtered.size().getInfo())
            print('\n')
            
            return s2_filtered 

        # function for masking non-vegetation areas
        def maskS2nonvegetation(image):

                qa    = image.select('QA60')
                scl   = image.select('SCL')
                ndvi  = image.select('NDVI')
                mndvi = image.select('MNDVI')

                cloudBitMask = 1 << 10
                cirrusBitMask = 1 << 11

                #vegetationMask1 = 4 # vegetation
                #vegetationMask2 = 5 # non-vegetated
                #vegetationMask3 = 6 # water
                #vegetationMask4 = 7 # unclassified
                #vegetationMask5 = 11 # snow

                # this mask selects vegetation + non-vegetated + water + unclassified + areas with VIs (NDVI and MNDVI) greater that a threshold set in the configuration file
                #mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).And(qa.bitwiseAnd(cloudBitMask).eq(0)).And(qa.bitwiseAnd(cirrusBitMask).eq(0)).And(ndvi.gte(ndviMask)).And(mndvi.gte(mndviMask))
                mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11)).And(qa.bitwiseAnd(cloudBitMask).eq(0)).And(qa.bitwiseAnd(cirrusBitMask).eq(0)).And(ndvi.gte(ndviMask)).And(mndvi.gte(mndviMask))
                #mask = scl.gte(4).And(qa.bitwiseAnd(cloudBitMask).eq(0)).And(qa.bitwiseAnd(cirrusBitMask).eq(0)).And(ndvi.gte(ndviMask)).And(mndvi.gte(mndviMask))
                #mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).And(mndvi.gte(0.05)) 

                vegetation = image.updateMask(mask)

                return vegetation

        # function to transform ee objects to dataframes pandas objects
        # function that transforms arrays into dataframes
        def ee_array_to_df(imagecollection, geometry, scale):

            """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""

            # select bands from the image collection
            filtered = imagecollection.select(bands)

            # function that produce functions to reduce a region with atatistics (mean, max, min, etc.)
            def create_reduce_region_function(geometry,
                                                reducer=ee.Reducer.mean(),
                                                scale=1000,
                                                crs=crs,
                                                bestEffort=True,
                                                maxPixels=1e13,
                                                tileScale=4):

                    def reduce_region_function(img):

                        stat = img.reduceRegion(
                            reducer=reducer,
                            geometry=geometry,
                            scale=scale,
                            crs=crs,
                            bestEffort=bestEffort,
                            maxPixels=maxPixels,
                            tileScale=tileScale)

                        return ee.Feature(geometry, stat).set({'millis': img.date().millis()})
                    return reduce_region_function

            # function to transfer feature properties to a dictionary.
            def fc_to_dict(fc):
                    prop_names = fc.first().propertyNames()
                    prop_lists = fc.reduceColumns(reducer=ee.Reducer.toList().repeat(prop_names.size()),selectors=prop_names).get('list')

                    return ee.Dictionary.fromLists(prop_names, prop_lists)

            # creating reduction function (reduce_VI is a function)
            reduce_VI = create_reduce_region_function(
                geometry= geometry, reducer=ee.Reducer.mean(), scale=10, crs= crs)

            # transform image collection into feature collection (tables)
            VI_stat_fc = ee.FeatureCollection(imagecollection.map(reduce_VI)).filter(
                ee.Filter.notNull(imagecollection.first().bandNames()))

            # transform feature collection into dictionary object
            VI_dict = fc_to_dict(VI_stat_fc).getInfo()

            #print(type(VI_dict), '\n')

            #for prop in VI_dict.keys():
            #    print(prop + ':', VI_dict[prop][0:3] + ['...'])

            # transform dictionary into dataframe
            VI_df = pd.DataFrame(VI_dict)

            # convert column in datatime type object
            #VI_df['datetime'] = pd.to_datetime(VI_df['time'], unit='ms')
            VI_df['date']     = pd.to_datetime(VI_df['millis'], unit='ms').dt.date

            # generate a list with the names of each band of the collection 
            list_of_bands = filtered.first().bandNames().getInfo()

            # remove rows without data inside.
            VI_df = VI_df[['date', *list_of_bands]].dropna()

            # convert the data to numeric values.
            for band in list_of_bands:
                VI_df[band] = pd.to_numeric(VI_df[band], errors='coerce', downcast ='float')

            # convert the time field into a datetime.
            #VI_df['datetime'] = pd.to_datetime(VI_df['time'], unit='ms')
            #VI_df['date']     = pd.to_datetime(VI_df['time'], unit='ms').dt.date

            # keep the columns of interest.
            #VI_df = VI_df[['datetime','date',  *list_of_bands]]

            # flag to identify if in the reduction there were pixels, or they were masked-removed
            VI_df['flag'] = 100

            # reduction in case there are two pixels from different images for the same day
            VI_df = VI_df.groupby('date').mean().reset_index().set_index('date').copy()

            return VI_df

        # applying functions 

        # request of catalogues 
        S2     = load_catalog(COPERNICUS_S2_L2A, timeSD, point, COPERNICUS_S2_bands)
        temp   = load_catalog(MODIS_temp,        timeSD, point, MODIS_temp_bands)
        prec   = load_catalog(USAID_prec,        timeSD, point, USAID_prec_bands)
        gpp_MODIS    = load_catalog(MODIS_GPP,         timeSD, point, MODIS_GPP_bands)
        npp_MODIS    = load_catalog(MODIS_NPP,         timeSD, point,  MODIS_NPP_bands)

        # calculation of vegetation indices for the collection
        S2_VI = S2.map(calculateVI)

        # filter cloud coverage
        cloud_coverage_metadata_name = 'CLOUDY_PIXEL_PERCENTAGE'                     # name of metadata property indicating cloud coverage in %

        # applying cloud filter 
        S2_VI = cloud_filter(S2_VI, cloud_coverage_metadata_name, max_cloud_coverage)   # max cloud coverage defined in the Config file

        # apply cloud local filter
        # S2_VI = local_cloud_filter(S2_VI, aoi, max_cloud_coverage)

        # applying mask 
        S2_VI = S2_VI.map(maskS2nonvegetation)

        # select bands for the analysis of interdependency and regression models 
        #bands = ['NDVI','EVI','EVI2','CLr','CLg','MTCI','MNDVI','MNDWI','LSWI','NDII']
        # bands = ['NDVI','EVI','EVI2','CLr','MNDVI','MNDWI','LSWI','NDII', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

        # applying funtion ee_array_to_df to the defined geometry
        S2_VI_df_aux  = ee_array_to_df(S2_VI, aoi, 10) 
        S2_VI_df_aux = S2_VI_df_aux.groupby(S2_VI_df_aux.index).sum().rename_axis('date')


        # loop to join each S2_VI_df in a single dataframe per year named as df_VI
        df_VI = pd.DataFrame(columns=bands)


        def add_date_info(df):

            df = df.reset_index()
            df['date'] = pd.to_datetime(df['date'])
            df['Year'] = pd.DatetimeIndex(df['date']).year

            def timestamp(df):
                df = df.timestamp()
                return df

            df['timestamp'] = df['date'].apply(timestamp)
            df = df.set_index('date')

            return df

        df_VI  = add_date_info(S2_VI_df_aux).rename_axis('date')

        # filtering VI_eshape by flag. Removes dates when there were areas whitin the climatological footprint that were totally masked 
        df_VI_filtered = df_VI.copy()
        df_VI_filtered = df_VI_filtered[df_VI_filtered['flag']>80].drop(['flag'], axis = 1)

        # create time series with daily frequency
        time_series = pd.date_range(start=start, end=end, freq="D")
        time_series = pd.DataFrame(time_series).rename(columns={0: 'date'}).set_index('date')

        # allows to have a time series with daily frequency with gaps when the VI were not calculated or there were not S2 images
        df_VI_time = pd.merge(left= time_series, right = df_VI_filtered,
                                        how="left", left_index = True , right_index = True)  

        # interpolate values
        df_VI_interpolate = df_VI_time.interpolate(method='akima', order=1, limit_direction ='forward')
        #df_VI_interpolate_limits = df_VI_interpolate.apply(lambda x: x.interpolate(method="spline", order=6))
        #df_VI_interpolate = df_VI_interpolate.fillna(method='backfill')
        #df_VI_interpolate = df_VI_interpolate.fillna(method='ffill')

        # file to extrapolate
        df_VI_export = df_VI_interpolate.dropna().drop(['Year','timestamp'], axis = 1) 
        df_VI_export.to_csv(outputdir + '/VI_output/' + ID + "_Vegetation_indices.csv")   

        print("       Exporting: Vegetation_indices.csv")
        print('\n') 

        #---------------------------------------------------------------------------------------------
        # Save plots of VI
        def plot_timeseries_vi_multiple(df):
            # subplots.
            fig, ax = plt.subplots(figsize=(14, 6)) #Indicates the size of the plot

            if 'NDVI' in bands:
                # add scatter plots //Adds the scatter points
                ax.plot(df['NDVI'],
                            c='#00FF00', alpha=1, label='NDVI', lw=2, linestyle = ':')
            if 'EVI' in bands:
                ax.plot(df['EVI'],
                            c='red', alpha=1, label='EVI', lw=2, linestyle = ':')
            if 'EVI2' in bands:
                ax.plot(df['EVI2'],
                            c='yellow', alpha=1, label='EVI-2', lw=2, linestyle = ':')
            if 'CLr' in bands:
                ax.plot(df['CLr'],
                            c='green', alpha=1, label='CLr', lw=2, linestyle = ':')
            if 'MNDVI' in bands:
                ax.plot(df['MNDVI'],
                            c='black', alpha=1, label='MNDVI', lw=2, linestyle = ':')
            if 'MNDWI' in bands:
                ax.plot(df['MNDWI'],
                            c='#00FFFF', alpha=0.5, label='MNDWI', lw=2, linestyle = '-.')
            if 'LSWI' in bands:
                ax.plot(df['LSWI'],
                            c='blue', alpha=0.8, label='LSWI', lw=2, linestyle = '-.')
            if 'NDII' in bands:
                ax.plot(df['NDII'],
                            c='#00008B', alpha=0.8, label='NDII', lw=2, linestyle = '-.') #marker="x", markersize=2)

            ax.set_title('Vegetation Indices', fontsize=16)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('Vegetation Index', fontsize=14)
            ax.set_ylim(-1, 1)
            ax.grid(lw=0.5)
            ax.legend(fontsize=14, loc='lower right')

            # shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, .8))
            #plt.savefig(outputdir + '/VI_output/' + ID + '_VI_timeseries.png', dpi=300, format='png', bbox_inches='tight',pad_inches=0.0001)
            plt.savefig(outputdir + '/VI_output/' + ID + '_VI_timeseries.png', dpi=300)

            return plt #.show()

        plot_timeseries_vi_multiple(df_VI_export)
        return df_VI_export

    def read_VI(self, outputdir, ID):

        df_vi_name = outputdir + '/VI_output/' + ID + "_Vegetation_indices.csv"  
        parser = lambda date: dt.datetime.strptime(date, "%Y-%m-%d")                               
                                                                                                                                                                     
        df_VI_export = pd.read_csv(df_vi_name, parse_dates=[0], 
                         date_parser=parser, index_col=0, header=0)
        
        return df_VI_export
    
