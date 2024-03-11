import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from t2listing import *
from t2data import *
from t2incons import *
from t2grids import *
import glob
# Set up steam tables
from pyXSteam.XSteam import XSteam
steam_table = XSteam(XSteam.UNIT_SYSTEM_MKS)

# Parent directory containing all data
parent_dir = 'C:/Users/arctic_vm1/Documents/PetraSim models/Qinghe/'
model_version = 'Qinghe_V09PS'
# Model version
# TODO modify to latest model version

# Load FOFT and GOFT Files
foft_name = parent_dir + model_version+'/FOFT'                  # FOFT file
foft_csv_name = parent_dir + model_version+'/foft.csv'          # foft.csv file
goft_name = parent_dir + model_version+'/GOFT'                  # GOFT file
goft_csv_name = parent_dir + model_version+'/goft.csv'          # goft.csv file
data_file_name = parent_dir + model_version+'/'+model_version+'.dat'   # TOUGH2 .dat file
#### plot tempreature and drawdown profile through the project lifetime
dir_processed_xlsx_wells = parent_dir + model_version+'/Processed_wells'   # Directory path to store processed data
dir_water_level_data = parent_dir+'Water level for Qinghe wells'                        # Directory path that stores water level measurements
wells_w_water_level = [filename.split('.')[0] for filename in os.listdir(dir_water_level_data) if os.path.splitext(os.path.join(dir_water_level_data, filename))[1]=='.xlsx']   # List of well long names with water level measurements
df_long_names = pd.read_csv(parent_dir+'Coordinate of wellheads GCS_CN_2000.csv')       # Dataframe with wellhead coordinates, and long and short well names
df_production_data = pd.read_csv(parent_dir+'Production/2018-2020 Daily Production wells total.csv') # Dataframe with production data for all wells
wells_w_prod_data = df_production_data['Well name'].unique()
prod_power_MW = [(row['Flow rate kg/s']*(steam_table.h_tx(row['Temp degreeC'], 0)-134))/1000 for idx,row in df_production_data.iterrows()]
df_production_data['Power MW'] = prod_power_MW
df_injection_data = pd.read_excel(parent_dir+'Production/2018-2020 Daily Reinjection wells total.xlsx') # Dataframe with reinjection data for all wells
wells_w_inj_data = df_injection_data['Well name'].unique()

# Import TOUGH2 data and history files (foft, goft)
dat = t2data(data_file_name)
foft = t2historyfile(foft_name)
foft_csv = pd.read_csv(foft_csv_name)
# Rename columns
foft_csv.rename(columns={'KCYC                ':'KCYC','TIME (s)            ':'TIME (s)','ELEM                ':'ELEM','INDEX               ':'INDEX',
                         'P (Pa)              ':'P (Pa)','T (deg C)           ':'T (deg C)', 'Sg                  ':'Sg','X(water2)           ':'X(water2)'}, errors="raise", inplace=True)
# Modify 'ELEM' values to get only last 5 chars of string
for idx, row in foft_csv.iterrows():
    foft_csv.at[idx, 'ELEM'] = row['ELEM'][-5:]

goft = t2historyfile(goft_name)
print('------------MODEL VERSION: '+model_version+'------------\ndat, foft and goft files loaded')
print('Number of blocks: '+str(dat.grid.num_blocks))
goft_csv = pd.read_csv(goft_csv_name)
goft_csv.rename(columns={'KCYC                ':'KCYC', 'TIME (s)            ':'TIME (s)', 'ELEM                ':'ELEM', 'INDEX               ':'INDEX',
                         'SOURCE              ':'SOURCE','S_INDEX             ':'S_INDEX', 'RATE (kg/s)         ':'RATE (kg/s)','ENTHALPY (J/kg)     ':'ENTHALPY (J/kg)', 'COL3                ':'COL3',
                         'FLOW FRACTION (GAS) ':'FLOW FRACTION (GAS)', 'Pwell (Pa)          ':'Pwell (Pa)'}, errors="raise", inplace=True)
# Modify 'ELEM' values to get only last 5 chars of string
for idx, row in goft_csv.iterrows():
    goft_csv.at[idx, 'ELEM'] = row['ELEM'][-5:]
    goft_csv.at[idx, 'SOURCE'] = row['SOURCE'][-5:]

# Rock types and their respective model layers
############ TODO NEEDS TO BE MODIFIED MANUALLY FOR EACH MODEL
rocktype_layer_n = {'QP001':1, 'QP002':2, 'NM001':3, 'NM002':4, 'NM003':5, 'NM004':6, 'NM005':7, 'NM006':8, 'NM007':9, 'NG001':10, 'NG002':11, 'NG003':12, 'NG004':13, 'NG005':14, 'NG051':14, 'ED001':15, 'BOTT1':16}

# Get rock types and layer numbers for each element (done only for time 0, or KCYC 1, to avoid repetitive iterations)
df_elem_rock = foft_csv.loc[foft_csv['KCYC']==1].drop(['KCYC', 'TIME (s)', 'INDEX','P (Pa)','T (deg C)','Sg','X(water2)'], axis=1)
block_rocks = [str(blk.rocktype) for blk in dat.history_block if str(blk) in df_elem_rock['ELEM'].to_list()]
block_layer = [rocktype_layer_n[rck] for rck in block_rocks]

df_elem_rock['Rock'] = block_rocks
df_elem_rock['Layer'] = block_layer
# Modify 'ELEM' values to get only last 5 chars of string
for idx, row in df_elem_rock.iterrows():
    df_elem_rock.at[idx, 'ELEM'] = row['ELEM'][-5:]

# Get well indexes (names) and types
well_types = []     # Type of the well, either P or R
well_names = []     # Name of the well in elem block gotten from the generator list index
dict_gener_idx = dict(dat.generator.keys())

for blk in df_elem_rock['ELEM'].values:
    if blk in list(dict_gener_idx.keys()):
        well_names.append(dict_gener_idx.get(blk))
        if dat.generator[(blk, dict_gener_idx.get(blk))].type == 'MASS':
            well_types.append('P')
        else:
            well_types.append('R')
    else:
        well_names.append('Not producing')
        # well_names.append(dict_gener_idx.get(blk))
        well_types.append('None')

df_elem_rock['Type'] = well_types
df_elem_rock['Well'] = well_names

# New FOFT dataframe including rocktypes and layer number
foft_full = pd.merge(foft_csv, df_elem_rock, on='ELEM').drop(['INDEX', 'Sg', 'X(water2)'], axis=1)
foft_full['TIME (days)'] = foft_full['TIME (s)']/(24*3600)    # Convert time to days
foft_full.drop('TIME (s)', axis=1, inplace=True)
foft_full['P (bar)'] = foft_full['P (Pa)']/100000                  # Convert pressure to bar
foft_full.drop('P (Pa)', axis=1, inplace=True)
# Save csv with the processed FOFT file
foft_full.to_csv('./'+model_version+'/foft_processed.csv')
print('FOFT Processed and saved.')
# New GOFT dataframe including rocktypes and layer number
goft_full = pd.merge(goft_csv, df_elem_rock, on='ELEM').drop(['S_INDEX', 'INDEX', 'FLOW FRACTION (GAS)', 'Pwell (Pa)'], axis=1)
goft_full['TIME (days)'] = goft_full['TIME (s)']/(24*3600)    # Convert time to days
goft_full.drop('TIME (s)', axis=1, inplace=True)
goft_full['RATE (kg/s)'] = goft_full['RATE (kg/s)']*-1    # Multiply rate by -1
goft_full['ENTHALPY (kJ/kg)'] = goft_full['ENTHALPY (J/kg)']/1000 
goft_full.drop('ENTHALPY (J/kg)', axis=1, inplace=True)
goft_full['POWER (MW)'] = goft_full['RATE (kg/s)'] * (goft_full['ENTHALPY (kJ/kg)']-134)/1000
# Save csv with the processed GOFT file
goft_full.to_csv('./'+model_version+'/goft_processed.csv')
print('GOFT Processed and saved.')
# Save xlsx with well codes and elem ids to manually find well names
well_names_code = pd.DataFrame(columns=['Well_code'])
well_names_code['Well_code'] = list(foft_full['Well'].unique())
unique_elem_list = []
for well_code in well_names_code['Well_code']:
    unique_elem_list.append(foft_full.loc[foft_full['Well']==well_code]['ELEM'].unique()[0])
well_names_code['ELEM'] = unique_elem_list
well_names_code.to_excel('./'+model_version+'/well_names_codes.xlsx') # Save excel with well codes (indexes) and block names

#######################################################################################
######################################## BREAK ########################################
#######################################################################################

# If running for the first time, run until here, get an error and start modifying a new
# excel file called 'Well names and codes.xlsx' in the parent dir with the well codes from 
# the foft and relate them to their well short names on PetraSim

# File with well names and codes (manually made)
# TODO Copy and paste codes from well_names_codes.xlsx into new excel and manually insert names into new column 'Well_name'
df_well_names_codes = pd.read_excel(parent_dir + model_version+'/Well names and codes.xlsx')#, converters={'Well_code':str})

# List of names of heating stations
station_code_list = []
for name in df_well_names_codes.loc[df_well_names_codes['Well_code']!='Not producing']['Well_name'].to_list():
    if name[:-1] not in station_code_list:
        station_code_list.append(name[:-1])

# Compute Darcy-Weisbach equation for P loss due to friction in the pipe
def darcy_weisbach_P_loss(L, T_prod, P_avg, m_rate):
    Dh = 0.1778
    A = np.pi * (Dh/2)**2 
    # fd = 0.0509555    # Darcy firction factor for Re 500.000, surface roughness 0.005 m, and 0.1778 m diam pipe
    # fd = 0.022          # https://www.sciencedirect.com/science/article/pii/S0375650520301930
    fd = 0.0327   # Darcy firction factor for Re 1.000.000, surface roughness 0.00177 m, and 0.1778 m diam pipe

    # L = P_avg*100000/(9.81*steam_table.rho_pt((P_avg-1)/2,T_prod))
    # L = 1000
    # Pressure loss in Pa
    P_loss = L * fd * steam_table.rho_pt(P_avg, T_prod)/2 * (m_rate /(A*steam_table.rho_pt(P_avg, T_prod)))**2 /Dh#
    return P_loss*1e-5

#### PCP_measured_calc to compute the Pressure Control Point from water level drawdown
def PCP_measured(well_long_name, water_level, water_level_init, z_PCP, T_prod, m_prod):
    well_short_name = str(df_long_names.loc[df_long_names['Well name']==well_long_name]['Well short name'].values[0])
    # Initial pressre is atmospheric (1 bar)
    z_pump = 160 # Pump depths in Qinghe are normally 160 m (below surface)
    P0 = 1.01325 # (bar)
    P_Grad = 0.1    # (bar/m)
    P_z_pump = P0 + P_Grad*z_pump  # Initial pressure at the pump depth
    
    # L = distance between PCP and pump to estimate friction pressure loss 
    L = z_PCP-(z_pump + water_level_init)
    p_loss_DW = darcy_weisbach_P_loss(L, T_prod, P_z_pump, m_prod)
    p_iter = P0 + 12e-5*P_Grad*water_level_init
    # Array with n divisions between measured water level and PCP depth
    n_divs = 1000.
    delta_h = (z_PCP-water_level_init)/n_divs
    h_iter = water_level
    # Iterate to depth of the PCP and compute P
    while h_iter<=z_PCP:
        if h_iter==z_PCP:
            print(well_long_name, h_iter)
        p_iter+= steam_table.rho_pt(p_iter, T_prod) * 9.8 * delta_h * 1e-5 
        h_iter+= delta_h
        
    # Account for P loss
    p_iter = p_iter + p_loss_DW
    return p_iter, p_loss_DW

#### T_P_DD_prod_well to calculate the average temperature, average pressure and water level drawdown
#   in: df_foft a dataframe containing foft data from the model output
#       df_goft a dataframe containing goft data from the model output
#   out: write an excel file with df_well_summary that contains
#       - Avg temperature
#       - Avg pressure
#       - Total mass flow
#       - Total power
#       - Water level difference
def T_P_DD_prod_well(df_foft, df_goft):
    # Filter production and injection wells
    df_foft = df_foft.loc[df_foft['Type']!='None']
    df_goft = df_goft.loc[df_goft['Type']!='None']
    well_codes = list(df_foft['Well'].unique())

    for well_code in well_codes:
        # Get well name from excel file with codes and names
        well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
        # well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])

        df_well_summary = pd.DataFrame()
        # Filter foft and goft dataframes with specific well data
        foft_sub_df = df_foft.loc[df_foft['Well']==well_code].copy()
        goft_sub_df = df_goft.loc[df_goft['Well']==well_code].copy()

        # Combine the ELEM and Layer columns in FOFT and GOFT
        foft_sub_df.loc[foft_sub_df['KCYC']>0, 'ELEM_Layer'] = foft_sub_df['ELEM'] + ':' + foft_sub_df['Layer'].astype(str)
        foft_sub_df.drop(['KCYC', 'ELEM','Layer', 'Well', 'Rock', 'Type'], axis=1, inplace=True)
        goft_sub_df.loc[goft_sub_df['KCYC']>0, 'ELEM_Layer'] = goft_sub_df['ELEM'] + ':' + goft_sub_df['Layer'].astype(str)
        goft_sub_df.drop(['KCYC', 'ELEM','Layer', 'Well', 'Rock', 'Type'], axis=1, inplace=True)

        # Unpivot T and P values
        foft_sub_df_unpivot = pd.melt(foft_sub_df, id_vars=['TIME (days)', 'ELEM_Layer'], value_vars=['T (deg C)', 'P (bar)'])
        foft_sub_df_unpivot.loc[foft_sub_df_unpivot['TIME (days)']>=0, 'ELEM_Layer_Var'] = foft_sub_df_unpivot['ELEM_Layer'] + ' - ' + foft_sub_df_unpivot['variable'].astype(str)
        foft_sub_df_unpivot.drop(['ELEM_Layer', 'variable'], axis=1, inplace=True)

        goft_sub_df_unpivot = pd.melt(goft_sub_df, id_vars=['TIME (days)', 'ELEM_Layer'], value_vars=['RATE (kg/s)', 'ENTHALPY (kJ/kg)', 'POWER (MW)'])
        goft_sub_df_unpivot.loc[goft_sub_df_unpivot['TIME (days)']>=0, 'ELEM_Layer_Var'] = goft_sub_df_unpivot['ELEM_Layer'] + ' - ' + goft_sub_df_unpivot['variable'].astype(str)
        goft_sub_df_unpivot.drop(['ELEM_Layer', 'variable'], axis=1, inplace=True)

        # Pivot Elem:layer and variable
        foft_sub_df_pivot = foft_sub_df_unpivot.pivot_table('value', 'TIME (days)', 'ELEM_Layer_Var')
        goft_sub_df_pivot = goft_sub_df_unpivot.pivot_table('value', 'TIME (days)', 'ELEM_Layer_Var')

        # Lists with the name of the columns for each variable
        T_columns = [col for col in foft_sub_df_pivot.columns if 'T (deg C)' in col]            # Temperature columns
        P_columns = [col for col in foft_sub_df_pivot.columns if 'P (bar)' in col]              # Pressure columns
        H_columns = [col for col in goft_sub_df_pivot.columns if 'ENTHALPY (kJ/kg)' in col]     # Enthalpy columns
        Pow_columns = [col for col in goft_sub_df_pivot.columns if 'POWER (MW)' in col]         # Power columns
        M_columns = [col for col in goft_sub_df_pivot.columns if 'RATE (kg/s)' in col]          # Mass rate columns
        # Calculate PCP depth from production blocks depths and permeabilities
        perm_xy = [dat.grid.block[col_name[:5]].rocktype.permeability[0] for col_name in P_columns]                # List of elem XY permeability values
        prod_blk_depths = {col_name:-dat.grid.block[col_name[:5]].centre[2] for col_name in P_columns}             # Dictionary with P column names as keys and block depths as value
        
        z_PCP = np.average(list(prod_blk_depths.values()), weights=perm_xy)
        z_PCP_prev = 0
        z_PCP_post = 0
        # Get the block names surrounding the PCP to interpolate the pressure at the PCP depth
        for i in range(len(list(prod_blk_depths.values()))):
            if z_PCP < list(prod_blk_depths.values())[i] and  z_PCP >=  list(prod_blk_depths.values())[i+1]:
                z_PCP_post+=  list(prod_blk_depths.values())[i]             # Update depth of lower element
                P_col_name_post = [k for k,v in prod_blk_depths.items() if v ==z_PCP_post][0] # Update pressure column name of lower element
                z_PCP_prev+=  list(prod_blk_depths.values())[i+1]           # Update depth of upper element
                P_col_name_prev = [k for k,v in prod_blk_depths.items() if v ==z_PCP_prev][0] # Update pressure column name of upper element
                break

        z_mean_PCP_all_wells = 1246.2   # Calculated as the average PCP of the 4 wells. Done manually after processing initially 1246.2
        z_all_PCP_prev = 0
        z_all_PCP_post = 0
        # Get the block names surrounding the PCP to interpolate the pressure at the PCP depth
        for j in range(len(list(prod_blk_depths.values()))):
            if z_mean_PCP_all_wells < list(prod_blk_depths.values())[j] and  z_mean_PCP_all_wells >=  list(prod_blk_depths.values())[j+1]:
                z_all_PCP_post+=  list(prod_blk_depths.values())[j]             # Update depth of lower element
                P_col_name_post_all = [k for k,v in prod_blk_depths.items() if v ==z_all_PCP_post][0] # Update pressure column name of lower element
                z_all_PCP_prev+=  list(prod_blk_depths.values())[j+1]           # Update depth of upper element
                P_col_name_prev_all = [k for k,v in prod_blk_depths.items() if v ==z_all_PCP_prev][0] # Update pressure column name of upper element
                break
        ## Calculate well properties
        # Average pressure (normal average)
        df_well_summary['Avg Pressure (bar)'] = foft_sub_df_pivot[P_columns].mean(axis=1)
        # Average temperature and Average enthalpy (weighted average with mass flow rate)
        average_T = []
        average_PCP = []    # Average pressure at the PCP depth in each well
        z_PCP_list = []

        average_Pressure_mean_z = []    # Average pressure at the 4well-average PCP arbitrary depth
        z_all_PCP_list = []
        col_name_prev_all_list = []

        # Iteration to calculate average T which are weighted averages
        for idx, row in foft_sub_df_pivot.iterrows():
            # Filter wells in which foft and goft cells do not match (problem source not known)
            # if len(goft_sub_df_pivot.loc[idx][M_columns].values)==len(row[T_columns].values):
            if sum(goft_sub_df_pivot.loc[idx][M_columns].values)==0:
                average_T.append(np.average(row[T_columns].values))
                average_PCP.append(((z_PCP-z_PCP_prev)*(row[P_col_name_post] - row[P_col_name_prev]) /(z_PCP_post-z_PCP_prev)) + row[P_col_name_prev])
                z_PCP_list.append(z_PCP)

                # Pressure at the arbitrary depth of 4well-average PCP depth
                average_Pressure_mean_z.append(((z_mean_PCP_all_wells-z_all_PCP_prev)*(row[P_col_name_post_all] - row[P_col_name_prev_all]) /(z_all_PCP_post-z_all_PCP_prev)) + row[P_col_name_prev_all])
                z_all_PCP_list.append(z_mean_PCP_all_wells)
                col_name_prev_all_list.append('')
            else:
                average_T.append(np.average(row[T_columns].values, weights=goft_sub_df_pivot.loc[idx][M_columns].values))
                # Calculate interpolated pressure of Pressure Control Point
                average_PCP.append(((z_PCP-z_PCP_prev)*(row[P_col_name_post] - row[P_col_name_prev]) /(z_PCP_post-z_PCP_prev)) + row[P_col_name_prev])
                z_PCP_list.append(z_PCP)
                # Pressure at the arbitrary depth of 4well-average PCP depth
                average_Pressure_mean_z.append(((z_mean_PCP_all_wells-z_all_PCP_prev)*(row[P_col_name_post_all] - row[P_col_name_prev_all]) /(z_all_PCP_post-z_all_PCP_prev)) + row[P_col_name_prev_all])
                z_all_PCP_list.append(z_mean_PCP_all_wells)
                col_name_prev_all_list.append(P_col_name_prev_all)
            # else:
            #     average_T.append(0.)
            #     average_PCP.append(0.)
            #     z_PCP_list.append(0.)
        df_well_summary['Avg Temperature (deg C)'] = average_T
        df_well_summary['Avg PCP (bar)'] = average_PCP
        df_well_summary['PCP depth (m)'] = z_PCP_list

        df_well_summary['P arbitrary (bar)'] = average_Pressure_mean_z
        df_well_summary['Arbitrary depth (m)'] = z_all_PCP_list
        df_well_summary['P Col Name Prev All'] = col_name_prev_all_list
        
        drawdown = []
        drawdown_PCP = []
        for idx, row in df_well_summary.iterrows():
            if row['Avg Temperature (deg C)'] > 0:
                # Water level drawdown, converting bar into PA (1e5)
                drawdown.append((df_well_summary['Avg Pressure (bar)'].iloc[0] - row['Avg Pressure (bar)'])*(-1e5)/(9.81*steam_table.rho_pt((row['Avg Pressure (bar)']), row['Avg Temperature (deg C)'])) )
                drawdown_PCP.append((df_well_summary['Avg PCP (bar)'].iloc[0] - row['Avg PCP (bar)'])*(-1e5)/(9.81*steam_table.rho_pt((row['Avg PCP (bar)']), row['Avg Temperature (deg C)'])) )

                # drawdown.append((row['Avg Pressure (bar)'])*100000/(9.81*steam_table.rho_pt((row['Avg Pressure (bar)']-1)/2,row['Avg Temperature (deg C)'])) )
                # drawdown_PCP.append((row['Avg PCP (bar)'])*100000/(9.81*steam_table.rho_pt((row['Avg PCP (bar)']-1)/2,row['Avg Temperature (deg C)'])) )
            else:
                drawdown.append(np.nan)
                drawdown_PCP.append(np.nan)
        df_well_summary['Drawdown (m)'] = drawdown
        df_well_summary['Drawdown PCP (m)'] = drawdown_PCP
        # Total mass flow rate
        df_well_summary['Total Rate (kg/s)'] = goft_sub_df_pivot[M_columns].sum(axis=1)
        # Total power
        df_well_summary['Total Power (MW)'] = goft_sub_df_pivot[Pow_columns].sum(axis=1)

        # Save well summary dataframe with avg P, T and Drawdown
        df_well_summary.to_excel(dir_processed_xlsx_wells+'/'+well_name+'_'+well_code+'.xlsx')
    
    print('All wells have been parsed and their data have been calculated.')
    print('Total timesteps:', len(df_well_summary['Avg Temperature (deg C)']))



#### plot_PCP_profiles to plot PCP graphics of modelled PCP and measured waterl if available
#   in: filename of the excel file output from the main function T_P_DD_prod_well
#   out: plot graphs
def plot_PCP_profiles(dic_colors, plot_measure:bool, station_PS_name:str, station_list):
    # Generate plots for each well station in the station_list
    for station_name in station_list:
        file_station_name_pattern = f"*{station_name}*.xlsx" # Files must contain the name of the station where the pressure signal test is carried out
        filtered_station_files = glob.glob(f"{dir_processed_xlsx_wells}/{file_station_name_pattern}")

        # Iterate through the filtered files to plot pressure signal tests
        plt.figure(figsize=(15,10))
        for file_path in filtered_station_files:
            well_code = os.path.splitext(os.path.basename(file_path))[0].split('_')[1]
            well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
            well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])
            well_type = df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Type'].values[0]
            well_model_df = pd.read_excel(file_path).interpolate(method='polynomial', order=2)
            
            z_PCP = well_model_df['PCP depth (m)'].mean()
            # Plot field water level data if available
            if plot_measure and well_long_name in wells_w_water_level:
                # Read excel file with water level data
                f_water = os.path.join(dir_water_level_data, well_long_name+'.xlsx')
                well_water_df = pd.read_excel(f_water, sheet_name='Water Level')
                if well_long_name in wells_w_prod_data:
                    T_prod = np.mean(df_production_data.loc[df_production_data['Well name']==well_long_name]['Temp degreeC'])
                    m_prod = np.mean(df_production_data.loc[df_production_data['Well name']==well_long_name]['Flow rate kg/s'])
                else:
                    T_prod = well_model_df['Avg Temperature (deg C)'].mean()
                    m_prod = well_model_df['Total Rate (kg/s)'].max()
                PCP_measured_list = []
                # df_test = pd.DataFrame(columns=['Year', 'Water'])
                # df_test['Year']=well_water_df['Time (years)']
                # df_test['Water']=well_water_df['Water level (m)']
                # print(df_test)
                for w_level in well_water_df['Water level (m)'].tolist():
                    PCP_measured_list.append(PCP_measured(well_long_name, w_level, z_PCP, T_prod, m_prod))
                # Plot measured PCP pressure calculated from the water level
                plt.plot((well_water_df['Time (years)']-well_water_df['Time (years)'][0])*365, PCP_measured_list, c=dic_colors[well_name], label='Measured PCP P '+well_name, linestyle = '--', marker='o')
                plt.plot(well_model_df['TIME (days)'], well_model_df['Avg PCP (bar)'], label=well_type+': '+well_name, c=dic_colors[well_name])
            else:
                # If the iteration station name is the interest Pressure signal station, apply proper symbology.
                if station_name == station_PS_name:
                    plt.plot(well_model_df['TIME (days)'], well_model_df['Avg PCP (bar)'], label=well_type+': '+well_name, c=dic_colors[well_name])
                else:
                    plt.plot(well_model_df['TIME (days)'], well_model_df['Avg PCP (bar)'], label=well_name)
        plt.xlabel('Time (days)')
        plt.ylabel('Pressure (bar)')
        plt.xlim((0,36))
        # plt.ylim((120,125))
        plt.grid(color='grey', linestyle = '--', linewidth = 0.5)
        for i in range(0,9):
            plt.axvline(x=4*i, c='black')
            plt.text(4*i+0.5, 108, 'P'+str(i))
        plt.title('Pressure at PCP depth - '+model_version)
        plt.legend()
        plt.savefig(dir_processed_xlsx_wells+'/'+station_name+'_PCP Graph.png', transparent=False)
        plt.close()

#### plot_mass_power to plot modelled mass and power output from wells
#   in: filename of the excel file output from the main function T_P_DD_prod_well
#   out: plot graphs
def plot_mass(dic_colors, plt_measured:bool, station_PS_name:str, station_list):
    file_station_name_pattern = f"*{station_PS_name}*.xlsx" # Files must contain the name of the station where the pressure signal test is carried out
    filtered_station_files = glob.glob(f"{dir_processed_xlsx_wells}/{file_station_name_pattern}")

    # Iterate through the filtered files to plot pressure signal tests
    plt.figure(figsize=(15,10))
    for file_path in filtered_station_files:
        well_code = os.path.splitext(os.path.basename(file_path))[0].split('_')[1]
        well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
        well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])
        well_type = df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Type'].values[0]
        well_model_df = pd.read_excel(file_path).interpolate(method='polynomial', order=2)
        well_type = df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Type'].values[0]
        if well_long_name not in wells_w_prod_data:
            plt.plot(well_model_df['TIME (days)'], well_model_df['Total Rate (kg/s)'], label=well_type+': '+well_name, c=dic_colors[well_name])
        else:
            plt.plot(well_model_df['TIME (days)'], well_model_df['Total Rate (kg/s)'], label=well_type+': '+well_name, c=dic_colors[well_name])
            # if plt_measured:
            #     well_prod_df = df_production_data.loc[df_production_data['Well name']==well_long_name]
            #     plt.plot(well_prod_df['Year']*365, well_prod_df['Flow rate kg/s'], c=dic_colors[well_name], label='Measured flow rate '+well_name, linestyle = '--', marker='o')
    plt.xlabel('Time (days)')
    plt.ylabel('Mass flow rate (kg/s)')
    plt.xlim((0,36))
    plt.grid(color='grey', linestyle = '--', linewidth = 0.5)
    for i in range(0,9):
        plt.axvline(x=4*i, c='black')
        plt.text(4*i+0.5, 0, 'P'+str(i))
    plt.title('Pressure at PCP depth - '+model_version)
    plt.legend()
    plt.savefig(dir_processed_xlsx_wells+'/'+station_PS_name+'_Mass Graph.png', transparent=False)
    plt.close()

#### plot_P_arbitrary_depth to plot P graphics of modelled Pressure at the samedepth for all wells and measured waterl if available
#   in: filename of the excel file output from the main function T_P_DD_prod_well
#   out: plot graphs
def plot_P_arbitrary_depth(dic_colors, station_PS_name:str, station_list):
    # Generate plots for each well station in the station_list
    for station_name in station_list:
        file_station_name_pattern = f"*{station_name}*.xlsx" # Files must contain the name of the station where the pressure signal test is carried out
        filtered_station_files = glob.glob(f"{dir_processed_xlsx_wells}/{file_station_name_pattern}")
        # Iterate through the filtered files to plot pressure signal tests
        plt.figure(figsize=(15,10))
        for file_path in filtered_station_files:
            well_code = os.path.splitext(os.path.basename(file_path))[0].split('_')[1]
            well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
            well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])
            well_type = df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Type'].values[0]
            well_model_df = pd.read_excel(file_path).interpolate(method='polynomial', order=2)
            well_type = df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Type'].values[0]
            z_PCP = well_model_df['PCP depth (m)'].mean()
            z_mean_PCP_all_wells = well_model_df['Arbitrary depth (m)'].mean()
            P_col_name = well_model_df['P Col Name Prev All'].unique()

            # If the iteration station name is the interest Pressure signal station, apply proper symbology.
            if station_name == station_PS_name:
                plt.plot(well_model_df['TIME (days)'], well_model_df['P arbitrary (bar)'], label=well_type+': '+well_name , c=dic_colors[well_name]) # label: +' at '+P_col_name[1][:8]
            else:
                plt.plot(well_model_df['TIME (days)'], well_model_df['P arbitrary (bar)'], label=well_name)
        plt.xlabel('Time (days)')
        plt.ylabel('Pressure (bar)')
        plt.xlim((0,36))
        # plt.ylim((120,125))
        plt.grid(color='grey', linestyle = '--', linewidth = 0.5)
        for i in range(0,9):
            plt.axvline(x=4*i, c='black')
            plt.text(4*i+0.5, 108, 'P'+str(i))
        plt.title('Pressure at '+ str(z_mean_PCP_all_wells) +' depth - '+model_version)
        plt.legend()
        plt.savefig(dir_processed_xlsx_wells+'/'+station_name+' P at '+ str(z_mean_PCP_all_wells) +'m Graph.png', transparent=False)
        plt.close() 

#######################################################################################
########################### START CALCULATION AND PLOTTING ############################
#######################################################################################

def plot_total_profiles():
    # Dictionary with well name and color for each well
    dic_colors = {'TSL1':'blue', 'TSL2':'red', 'TSL3':'maroon', 'TSL4':'teal'}
    plt_measured = False
    plot_PCP_profiles(dic_colors, plt_measured, 'TSL', station_code_list)
    plot_mass(dic_colors, plt_measured, 'TSL', station_code_list)
    plot_P_arbitrary_depth(dic_colors, 'TSL', station_code_list)
    print('Plotted mass flow rate and PCP Pressure for all wells')

# Process data for all wells
# T_P_DD_prod_well(foft_full, goft_full)
# CALL PLOTTING FUNCTIONS
plot_total_profiles()