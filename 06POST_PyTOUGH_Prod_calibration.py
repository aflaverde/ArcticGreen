import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from t2listing import *
from t2data import *
from t2incons import *
from t2grids import *
# Set up steam tables
from pyXSteam.XSteam import XSteam
steam_table = XSteam(XSteam.UNIT_SYSTEM_MKS)
from datetime import datetime
import time

start_time = time.time()
print('Started post-processing at', datetime.now())

# Parent directory containing all data
parent_dir = 'C:/Users/arctic_vm1/Documents/PetraSim models/Qinghe/'
model_version = 'Qinghe_V09'
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

# Get rock types and layer numbers for each element (done only for time 0 to avoid repetitive iterations)
df_elem_rock = foft_csv.loc[foft_csv['KCYC']==1].drop(['KCYC', 'TIME (s)', 'INDEX','P (Pa)','T (deg C)','Sg','X(water2)'], axis=1)
block_rocks = [str(blk.rocktype) for blk in dat.history_block]
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
        well_types.append('None')

df_elem_rock['Type'] = well_types
df_elem_rock['Well'] = well_names

# # Export dataframe with XY coordinates of NG004 elements to refine in later model
# df_elem_rock_xy_NG4 = df_elem_rock[df_elem_rock['Rock']=='NG004']
# elem_NG4_X_list = [dat.grid.block[row['ELEM']].centre[0] for idx, row in df_elem_rock_xy_NG4.iterrows()]
# elem_NG4_Y_list = [dat.grid.block[row['ELEM']].centre[1] for idx, row in df_elem_rock_xy_NG4.iterrows()]
# df_elem_rock_xy_NG4['X'] = elem_NG4_X_list
# df_elem_rock_xy_NG4['Y'] = elem_NG4_Y_list
# df_elem_rock_xy_NG4.to_excel(parent_dir+model_version+'/df_elem_rock_NG4.xlsx')
# print('Exported df elem rock')

# New FOFT dataframe including rocktypes and layer number
foft_full = pd.merge(foft_csv, df_elem_rock, on='ELEM').drop(['INDEX', 'Sg', 'X(water2)'], axis=1)
foft_full['TIME (years)'] = foft_full['TIME (s)']/(365*24*3600)    # Convert time to years
foft_full.drop('TIME (s)', axis=1, inplace=True)
foft_full['P (bar)'] = foft_full['P (Pa)']/100000                  # Convert pressure to bar
foft_full.drop('P (Pa)', axis=1, inplace=True)
# Save csv with the processed FOFT file
foft_full.to_csv('./'+model_version+'/foft_processed.csv')

# New GOFT dataframe including rocktypes and layer number
goft_full = pd.merge(goft_csv, df_elem_rock, on='ELEM').drop(['S_INDEX', 'INDEX', 'FLOW FRACTION (GAS)', 'Pwell (Pa)'], axis=1)
goft_full['TIME (years)'] = goft_full['TIME (s)']/(365*24*3600)    # Convert time to years
goft_full.drop('TIME (s)', axis=1, inplace=True)
goft_full['RATE (kg/s)'] = goft_full['RATE (kg/s)']*-1    # Multiply rate by -1
goft_full['ENTHALPY (kJ/kg)'] = goft_full['ENTHALPY (J/kg)']/1000   
goft_full.drop('ENTHALPY (J/kg)', axis=1, inplace=True)
goft_full['POWER (MW)'] = goft_full['RATE (kg/s)'] * (goft_full['ENTHALPY (kJ/kg)']-134)/1000
# Save csv with the processed GOFT file
goft_full.to_csv('./'+model_version+'/goft_processed.csv')

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
df_well_names_codes = pd.read_excel(parent_dir + model_version+'/Well names and codes.xlsx')

# Compute Darcy-Weisbach equation for P loss due to friction in the pipe
def darcy_weisbach_P_loss(L, T_prod, P_avg, m_rate):
    Dh = 0.1778
    A = np.pi * (Dh/2)**2 
    # fd = 0.0509555    # Darcy firction factor for Re 500.000, surface roughness 0.005 m, and 0.1778 m diam pipe
    # fd = 0.022          # https://www.sciencedirect.com/science/article/pii/S0375650520301930
    fd = 0.027   # Darcy firction factor for Re 2.000.000, surface roughness 0.0006 m, and 0.1778 m diam pipe
    #fd=0.01
    # Pressure loss in Pa
    P_loss = L * fd * steam_table.rho_pt(P_avg, T_prod)/2 * (m_rate /(A*steam_table.rho_pt(P_avg, T_prod)))**2 /Dh
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
    p_iter = P0
    # Array with n divisions between measured water level and PCP depth
    n_divs = 750.
    delta_h = (z_PCP-water_level_init)/n_divs
    h_iter = water_level
    # Iterate to depth of the PCP and compute P
    while round(h_iter,3)<=round(z_PCP,3):
        if h_iter==z_PCP:
            print(well_long_name, h_iter, delta_h)
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

    # Average PCP depth from all wells
    z_PCP_tot_average = 1242 # m

    for well_code in well_codes:
        # Get well name from excel file with codes and names
        well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
        
        well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])

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
        foft_sub_df_unpivot = pd.melt(foft_sub_df, id_vars=['TIME (years)', 'ELEM_Layer'], value_vars=['T (deg C)', 'P (bar)'])
        foft_sub_df_unpivot.loc[foft_sub_df_unpivot['TIME (years)']>=0, 'ELEM_Layer_Var'] = foft_sub_df_unpivot['ELEM_Layer'] + ' - ' + foft_sub_df_unpivot['variable'].astype(str)
        foft_sub_df_unpivot.drop(['ELEM_Layer', 'variable'], axis=1, inplace=True)

        goft_sub_df_unpivot = pd.melt(goft_sub_df, id_vars=['TIME (years)', 'ELEM_Layer'], value_vars=['RATE (kg/s)', 'ENTHALPY (kJ/kg)', 'POWER (MW)'])
        goft_sub_df_unpivot.loc[goft_sub_df_unpivot['TIME (years)']>=0, 'ELEM_Layer_Var'] = goft_sub_df_unpivot['ELEM_Layer'] + ' - ' + goft_sub_df_unpivot['variable'].astype(str)
        goft_sub_df_unpivot.drop(['ELEM_Layer', 'variable'], axis=1, inplace=True)

        # Pivot Elem:layer and variable
        foft_sub_df_pivot = foft_sub_df_unpivot.pivot_table('value', 'TIME (years)', 'ELEM_Layer_Var')
        goft_sub_df_pivot = goft_sub_df_unpivot.pivot_table('value', 'TIME (years)', 'ELEM_Layer_Var')

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
        z_PCP_prev_tot = 0
        z_PCP_post_tot = 0
        for i in range(len(list(prod_blk_depths.values()))):
            if z_PCP < list(prod_blk_depths.values())[i] and  z_PCP >=  list(prod_blk_depths.values())[i+1]:
                z_PCP_post+=  list(prod_blk_depths.values())[i]             # Update depth of lower element
                P_col_name_post = [k for k,v in prod_blk_depths.items() if v ==z_PCP_post][0] # Update pressure column name of lower element
                z_PCP_prev+=  list(prod_blk_depths.values())[i+1]           # Update depth of upper element
                P_col_name_prev = [k for k,v in prod_blk_depths.items() if v ==z_PCP_prev][0] # Update pressure column name of upper element
                break

            # elif z_PCP_tot_average < list(prod_blk_depths.values())[i] and  z_PCP_tot_average >=  list(prod_blk_depths.values())[i+1]:
            #     print('In tot if')
            #     z_PCP_post_tot+=  list(prod_blk_depths.values())[i]             # Update depth of lower element
            #     P_col_name_post_tot = [k for k,v in prod_blk_depths.items() if v ==z_PCP_post_tot][0] # Update pressure column name of lower element
            #     z_PCP_prev_tot+=  list(prod_blk_depths.values())[i+1]           # Update depth of upper element
            #     P_col_name_prev_tot = [k for k,v in prod_blk_depths.items() if v ==z_PCP_prev_tot][0] # Update pressure column name of upper element
            #     break
        ## Calculate well properties
        # Average pressure (normal average)
        df_well_summary['Avg Pressure (bar)'] = foft_sub_df_pivot[P_columns].mean(axis=1)
        # Average temperature and Average enthalpy (weighted average with mass flow rate)
        average_T = []
        average_PCP_P = []    # Average pressure at the PCP depth in each well
        average_PCP_tot_P = []  # Average 
        z_PCP_list = []
        z_PCP_tot_list = []   # List that contains the value of an arbitrary depth to calculate and compare pressure between wells

        # Iteration to calculate average T which are weighted averages
        for idx, row in foft_sub_df_pivot.iterrows():
            # Filter wells in which foft and goft cells do not match (problem source not known)
            if len(goft_sub_df_pivot.loc[idx][M_columns].values)==len(row[T_columns].values):
                if sum(goft_sub_df_pivot.loc[idx][M_columns].values)==0:
                    average_T.append(np.nan)
                    average_PCP_P.append(((z_PCP-z_PCP_prev)*(row[P_col_name_post] - row[P_col_name_prev]) /(z_PCP_post-z_PCP_prev)) + row[P_col_name_prev])
                    # average_PCP_tot_P.append(((z_PCP_tot_average-z_PCP_prev_tot)*(row[P_col_name_post_tot] - row[P_col_name_prev_tot]) /(z_PCP_post_tot-z_PCP_prev_tot)) + row[P_col_name_prev_tot])
                    z_PCP_list.append(z_PCP)
                    z_PCP_tot_list.append(z_PCP_tot_average)
                else:
                    average_T.append(np.average(row[T_columns].values, weights=goft_sub_df_pivot.loc[idx][M_columns].values))
                    # Calculate interpolated pressure of Pressure Control Point
                    average_PCP_P.append(((z_PCP-z_PCP_prev)*(row[P_col_name_post] - row[P_col_name_prev]) /(z_PCP_post-z_PCP_prev)) + row[P_col_name_prev])
                    # average_PCP_tot_P.append(((z_PCP_tot_average-z_PCP_prev_tot)*(row[P_col_name_post_tot] - row[P_col_name_prev_tot]) /(z_PCP_post_tot-z_PCP_prev_tot)) + row[P_col_name_prev_tot])
                    z_PCP_list.append(z_PCP)
                    z_PCP_tot_list.append(z_PCP_tot_average)
            # else:
            #     average_T.append(0.)
            #     average_PCP_P.append(0.)
            #     z_PCP_list.append(0.)
        df_well_summary['Avg Temperature (deg C)'] = average_T
        df_well_summary['Avg PCP (bar)'] = average_PCP_P
        # df_well_summary['Total avg PCP (bar)'] = average_PCP_tot_P
        df_well_summary['PCP depth (m)'] = z_PCP_list
        # df_well_summary['Total avg PCP depth (m)'] = z_PCP_tot_list
        
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

#### plot_T_profiles to plot temperature graphics of modelled T and measured WHT if available
#   in: filename of the excel file output from the main function T_P_DD_prod_well
#   out: plot graphs
def plot_T_profiles(f):
    well_code = os.path.splitext(os.path.basename(f))[0].split('_')[1]
    well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
    well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])
    # well_model_df = pd.read_excel(f).fillna(method='ffill')
    well_model_df = pd.read_excel(f).interpolate(method='polynomial', order=1)
    plt.figure(figsize=(15,10))
    # Plot temperature
    plt.plot(well_model_df['TIME (years)'], well_model_df['Avg Temperature (deg C)'], c='red', label='Modelled T')
    plt.xlabel('Time (years)')
    plt.xlim((0, max(well_model_df['TIME (years)'])))
    plt.ylim((20, 70))
    plt.ylabel('Temperature (°C)')

    # Plot field temperature data if available
    if well_long_name in wells_w_prod_data:
        plt.plot(df_production_data.loc[df_production_data['Well name']==well_long_name]['Year'], df_production_data.loc[df_production_data['Well name']==well_long_name]['Temp degreeC'], c='red', linestyle='--', label='Measured WHT', marker='+')

    # # Draw vertical lines
    # v_lines = list(range(1,30))
    # for xc in v_lines:
    #     plt.axvline(x=xc, color='#c9c9c9', linestyle='--')
    plt.grid(color='grey', linestyle = '--', linewidth = 0.5)
    plt.title(well_long_name +' - '+ well_name + ' Temperature')
    t_diff = round(well_model_df['Avg Temperature (deg C)'].dropna().tolist()[-1] - well_model_df['Avg Temperature (deg C)'].dropna().tolist()[0], 2)
    plt.text(20.5, 56, 'End of life Temp. difference: '+str(t_diff)+' deg C')
    plt.legend()
    plt.savefig(dir_processed_xlsx_wells+'/'+well_name+'_'+well_code+' - T Graph.png', transparent=False)
    plt.close()

#### plot_apparent_DD_profiles to plot drawdown graphics of modelled pressure and measured drawdown if available
#   in: filename of the excel file output from the main function T_P_DD_prod_well
#   out: plot graphs
def plot_apparent_DD_profiles(f):
    well_code = os.path.splitext(os.path.basename(f))[0].split('_')[1]
    # print(well_code)
    well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
    well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])
    
    # well_model_df = pd.read_excel(f).fillna(method='ffill')
    well_model_df = pd.read_excel(f).interpolate(method='polynomial', order=2)
    plt.figure(figsize=(15,10))

    z_PCP = well_model_df['PCP depth (m)'].mean()
    model_PCP_arr = np.array(well_model_df['Avg PCP (bar)'].tolist())
    initial_model_PCP = model_PCP_arr[np.logical_not(np.isnan(model_PCP_arr))][0]
    # Plot field water level data if available
    if well_long_name in wells_w_water_level:
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
        closest_to_t0_index = (well_water_df['Time (years)']-0).abs().idxmin()
        z_Water_level_init = well_water_df.loc[closest_to_t0_index, 'Water level (m)']
        for w_level in well_water_df['Water level (m)'].tolist():
            PCP_measured_list.append(PCP_measured(well_long_name, w_level, z_Water_level_init, z_PCP, T_prod, m_prod)[0])

        # Water level at time 0 is datum
        well_water_df['PCP_measured (bar)'] = PCP_measured_list
        P_PCP_t0 = well_water_df.loc[closest_to_t0_index, 'PCP_measured (bar)']

        # Plot measured PCP pressure drawdown calculated from the water level
        plt.scatter(well_water_df['Time (years)'], PCP_measured_list-P_PCP_t0, c='teal', label='Measured apparent drawdown')

        # Plot modelled drawdown
        plt.plot(well_model_df['TIME (years)'], well_model_df['Avg PCP (bar)']-initial_model_PCP, c='teal', label='Modelled apparent P drawdown', marker='+')

    else:
        # Plot only modelled drawdown
        # plt.plot(well_model_df['TIME (years)'], well_model_df['Drawdown (m)'], c='blue', label='Modelled water level difference')
        plt.plot(well_model_df['TIME (years)'], well_model_df['Avg PCP (bar)']-initial_model_PCP, c='teal', label='Modelled apparent P drawdown', marker='+')
    plt.xlabel('Time (years)')
    plt.ylabel('Apparent drawdown (bar)')
    plt.xlim((0, max(well_model_df['TIME (years)'])))
    plt.grid(color='grey', linestyle = '--', linewidth = 0.5)
    plt.title(well_long_name +' - '+ well_name + ' Apparent P Drawdown')
    plt.legend()
    plt.savefig(dir_processed_xlsx_wells+'/'+well_name+'_'+well_code+' - DD Graph.png', transparent=False)
    plt.close()

#### plot_PCP_profiles to plot PCP graphics of modelled PCP and measured waterl if available
#   in: filename of the excel file output from the main function T_P_DD_prod_well
#   out: plot graphs
def plot_absolute_PCP_profiles(f):
    well_code = os.path.splitext(os.path.basename(f))[0].split('_')[1]
    # print(well_code)
    well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
    well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])
    
    # well_model_df = pd.read_excel(f).fillna(method='ffill')
    well_model_df = pd.read_excel(f).interpolate(method='polynomial', order=2)
    plt.figure(figsize=(15,10))
    z_PCP = well_model_df['PCP depth (m)'].mean()
    # Plot field water level data if available
    if well_long_name in wells_w_water_level:
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

        closes_to_t0_index = (well_water_df['Time (years)']-0).abs().idxmin()
        z_Water_level_init = well_water_df.loc[closes_to_t0_index, 'Water level (m)']
        for w_level in well_water_df['Water level (m)'].tolist():
            PCP_measured_list.append(PCP_measured(well_long_name, w_level, z_Water_level_init, z_PCP, T_prod, m_prod)[0])
        # Plot measured PCP pressure calculated from the water level
        plt.scatter(well_water_df['Time (years)'], PCP_measured_list, c='teal', label='Measured PCP P')

        # Plot modelled pressure avg and PCP
        # plt.plot(well_model_df['TIME (years)'], well_model_df['Avg Pressure (bar)'], c='slateblue', label='Modelled Avg P')
        plt.plot(well_model_df['TIME (years)'], well_model_df['Avg PCP (bar)'], c='teal', label='Modelled PCP P')
        plt.xlabel('Time (years)')
        plt.xlim((0, max(well_model_df['TIME (years)'])))
        plt.ylabel('Pressure (bar)')

    else:
        # Plot only modelled pressure avg and PCP
        # plt.plot(well_model_df['TIME (years)'], well_model_df['Avg Pressure (bar)'], c='slateblue', label='Modelled Avg P')
        plt.plot(well_model_df['TIME (years)'], well_model_df['Avg PCP (bar)'], c='teal', label='Modelled PCP P')
        plt.xlabel('Time (years)')
        plt.xlim((0, max(well_model_df['TIME (years)'])))
        plt.ylabel('Pressure (bar)')

    plt.grid(color='grey', linestyle = '--', linewidth = 0.5)
    plt.title(well_long_name +' - '+ well_name + ' Absolute Pressure at PCP depth '+str(round(z_PCP, 2))+' m')
    plt.legend(loc='upper right')
    plt.savefig(dir_processed_xlsx_wells+'/'+well_name+'_'+well_code+' - PCP Graph.png', transparent=False)
    plt.close()




#### plot_PCP_measured to plot measured Pressure at the PCP depth calculated from the measured water level
#   in: filename of the excel file output from the main function T_P_DD_prod_well
#   out: plot graphs
def plot_DEPRECATEDPCP_measured(f):
    well_code = os.path.splitext(os.path.basename(f))[0].split('_')[1]
    # print(well_code)
    well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
    well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])
    
    # well_model_df = pd.read_excel(f).fillna(method='ffill')
    well_model_df = pd.read_excel(f).interpolate(method='polynomial', order=2)
    z_PCP = well_model_df['PCP depth (m)'].mean()
    # Plot field water level data if available
    if well_long_name in wells_w_water_level:
        fig, ax1 = plt.subplots(figsize=(15,10))
        ax2 = ax1.twinx()
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
        DW_Ploss_cumm = 0
        DW_point_count = 0
        for w_level in well_water_df['Water level (m)'].tolist():
            PCP_measuredP, DW_loss_WL = PCP_measured(well_long_name, w_level, z_PCP, T_prod, m_prod)
            PCP_measured_list.append(PCP_measuredP)
            # To calculate 
            DW_Ploss_cumm += DW_loss_WL
            DW_point_count +=1
        # Plot measured PCP pressure calculated from the water level
        ax1.scatter(well_water_df['Time (years)'], PCP_measured_list, c='teal', label='Measured PCP P')
        ax2.scatter(well_water_df['Time (years)'], -well_water_df['Water level (m)'], c='blue', label='Measured water level')

        plt.grid(color='grey', linestyle = '--', linewidth = 0.5)
        plt.title(well_long_name +' - '+ well_name + ' Pressure at PCP depth '+str(round(z_PCP, 2))+' m')
        fig.legend()
        plt.savefig(dir_processed_xlsx_wells+'/'+well_name+'_'+well_code+' - PCP measured.png', transparent=False)
        plt.close()
        DW_Ploss = DW_Ploss_cumm/DW_point_count
    else:
        DW_Ploss = 0
    return well_name, DW_Ploss

#### plot_mass_power to plot modelled mass and power output from wells
#   in: filename of the excel file output from the main function T_P_DD_prod_well
#   out: plot graphs
def plot_mass_power(f):
    well_code = os.path.splitext(os.path.basename(f))[0].split('_')[1]
    well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
    well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])
    well_model_df = pd.read_excel(f).fillna(method='ffill')

    # Twined x axis for secondary y axis with ax1: flow rate and ax2: power
    fig, ax1 = plt.subplots(figsize=(15,10))
    ax2 = ax1.twinx()
    # Plot field mass rate data if available
    if well_long_name in wells_w_prod_data:
        well_prod_df = df_production_data.loc[df_production_data['Well name']==well_long_name]
        # ax2.plot(well_prod_df['Year'], well_prod_df['Power MW'], c='grey', label='Measured power', marker='+')
        ax1.scatter(well_prod_df['Year'], well_prod_df['Flow rate kg/s'], c='green', label='Measured flow rate', marker='*')

        # Plot modelled mass rate and power
        ax1.plot(well_model_df['TIME (years)'], well_model_df['Total Rate (kg/s)'], c='green', label='Modelled flow rate')
        # ax2.plot(well_model_df['TIME (years)'], well_model_df['Total Power (MW)'], c='grey', label='Modelled power', linestyle = '--')

        ax1.set_xlabel('Time (years)')
        ax1.set_xlim(0,max(well_model_df['TIME (years)']))
        ax1.set_ylim(0,max(well_model_df['Total Rate (kg/s)'])+2)
        ax2.set_ylim(0,5)
        ax1.set_ylabel('Flow rate (kg/s)', color='green')
        ax1.tick_params(axis="y", labelcolor='green')
        ax2.set_ylabel('Thermal Power (MWth)')
        ax2.tick_params(axis="y", labelcolor='grey')
        ax1.xaxis.grid(True)
    else:
        # Plot only modelled flow rate and power
        ax1.plot(well_model_df['TIME (years)'], well_model_df['Total Rate (kg/s)'], c='green', label='Modelled flow rate')
        # ax2.plot(well_model_df['TIME (years)'], well_model_df['Total Power (MW)'], c='grey', label='Modelled power', linestyle = '--')

        ax1.set_xlabel('Time (years)')
        ax1.set_xlim(0,max(well_model_df['TIME (years)']))
        ax1.set_ylim(0,max(well_model_df['Total Rate (kg/s)'])+2)
        ax2.set_ylim(0,5)
        ax1.set_ylabel('Flow rate (kg/s)')
        ax1.set_ylabel('Flow rate (kg/s)', color='green')
        ax1.tick_params(axis="y", labelcolor='green')
        ax2.set_ylabel('Power (MW)')
        ax2.tick_params(axis="y", labelcolor='black')
        ax1.xaxis.grid(True)

    plt.title(well_long_name +' - '+ well_name + ' Mass flow rate and thermal power')
    fig.legend()
    plt.savefig(dir_processed_xlsx_wells+'/'+well_name+'_'+well_code+' - Power-Mass Graph.png', transparent=False)
    plt.close()

#### plot_DD_WLevel_profiles to plot drawdown graphics of modelled water level and measured drawdown if available
#   in: filename of the excel file output from the main function T_P_DD_prod_well
#   out: plot graphs
def plot_DD_WLevel_profiles(f):
    well_code = os.path.splitext(os.path.basename(f))[0].split('_')[1]
    # print(well_code)
    well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
    well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])
    
    # well_model_df = pd.read_excel(f).fillna(method='ffill')
    well_model_df = pd.read_excel(f).interpolate(method='polynomial', order=2)
    plt.figure(figsize=(15,10))

    # Plot field water level data if available
    if well_long_name in wells_w_water_level:
        # Read excel file with water level data
        f_water = os.path.join(dir_water_level_data, well_long_name+'.xlsx')
        well_water_df = pd.read_excel(f_water, sheet_name='Water Level')
        plt.plot(well_water_df['Time (years)'], -well_water_df['Water level (m)'], c='slateblue', linestyle='--', label='Measured water level')

        # Plot modelled drawdown
        # plt.plot(well_model_df['TIME (years)'], well_model_df['Drawdown (m)']-well_water_df['Water level (m)'].values[0], c='blue', label='Modelled water level')
        plt.plot(well_model_df['TIME (years)'], well_model_df['Drawdown PCP (m)']-well_water_df['Water level (m)'].values[0], c='teal', label='Modelled PCP water level')
        plt.xlabel('Time (years)')
        plt.ylabel('Water level (m)')

    else:
        # Plot only modelled drawdown
        # plt.plot(well_model_df['TIME (years)'], well_model_df['Drawdown (m)'], c='blue', label='Modelled water level difference')
        plt.plot(well_model_df['TIME (years)'], well_model_df['Drawdown PCP (m)'], c='teal', label='Modelled PCP water level difference')
        plt.xlabel('Time (years)')
        plt.ylabel('Water level difference (m)')

    plt.xlim((0, 4))
    plt.grid(color='grey', linestyle = '--', linewidth = 0.5)
    plt.title(well_long_name +' - '+ well_name + ' Water level drawdown')
    plt.legend()
    plt.savefig(dir_processed_xlsx_wells+'/'+well_name+'_'+well_code+' - DD Graph.png', transparent=False)
    plt.close()
#######################################################################################
########################### START CALCULATION AND PLOTTING ############################
#######################################################################################

def plot_total_profiles():
    # DW_loss_list = []
    # well_name_list = []
    # df_wells_DW_Ploss = pd.DataFrame()  # To save a dataframe with DW loss values
    # Iterate over xlsx files in folder of processed data
    for filename in os.listdir(dir_processed_xlsx_wells):
        f = os.path.join(dir_processed_xlsx_wells, filename)
        if os.path.splitext(f)[1]=='.xlsx':
            plot_apparent_DD_profiles(f)
            plot_T_profiles(f)
            # plot_mass_power(f)
            plot_absolute_PCP_profiles(f)

    #         well_name, DW_Ploss = plot_PCP_measured(f)
    #         well_name_list.append(well_name)
    #         DW_loss_list.append(DW_Ploss)
    # df_wells_DW_Ploss['Well_name'] = well_name_list
    # df_wells_DW_Ploss['DW P Loss'] = DW_loss_list
    # df_wells_DW_Ploss.to_excel('./DW Ploss.xlsx')
    print('Plotted temperature for all production wells.')
    print('Plotted drawdown for all production wells.')
    print('Plotted mass flow rate and power for all production wells.')

# Process data for all wells
T_P_DD_prod_well(foft_full, goft_full)
# CALL PLOTTING FUNCTIONS
plot_total_profiles()

end_time = time.time()
exec_time = end_time - start_time
print('Finished post-processing after', round(exec_time,2), 's')