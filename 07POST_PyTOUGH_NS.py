import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from t2listing import *
from t2data import *
from t2incons import *
from t2grids import *
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# Set up steam tables
from pyXSteam.XSteam import XSteam
steam_table = XSteam(XSteam.UNIT_SYSTEM_MKS)

# Parent directory containing all data
parent_dir = 'C:/Users/arctic_vm1/Documents/PetraSim models/Qinghe/' 
# Model version
# TODO modify to latest model version
model_version = 'Qinghe_V05NS'

# TOUGH2 .dat file path

data_file_name = parent_dir+ model_version+'/'+model_version+'.dat'   # TOUGH2 .dat file
# INCON/SAVE file path
save_file_name = parent_dir+ model_version+'/SAVE'  
foft_full_name = parent_dir+ model_version+'/foft_processed.csv'                  # foft_full.csv file from production calibration
df_long_names = pd.read_csv(parent_dir+'Coordinate of wellheads GCS_CN_2000.csv')       # Dataframe with wellhead coordinates, and long and short well names
# Test data folder
test_data_TVD_dir = parent_dir+'Testing data TVD/'
wells_w_test_data = [filename.split('.')[0][:-7] for filename in os.listdir(test_data_TVD_dir) if os.path.splitext(os.path.join(test_data_TVD_dir, filename))[1]=='.csv']   # List of well long names with water level measurements

# File with well names and codes (TODO manually made)
# TODO Copy and paste codes from well_names_codes.xlsx into new excel and manually insert names into new column 'Well_name'
df_well_names_codes = pd.read_excel(parent_dir+'Well names and codes.xlsx')   

# NS Calibration dir
ns_calibration_dir = parent_dir+ model_version+'/NS_calibration/'

foft_csv = pd.read_csv(foft_full_name)

# Import TOUGH2 data and incon/save files
dat = t2data(data_file_name)
inc = t2incon(save_file_name) # [0] Pressure, [1] Temperature

# Get well names from foft
foft_csv.sort_values(by='Well', ascending=True, inplace=True)
well_code_list = list(foft_csv['Well'].unique())
well_code_list.remove('Not producing')
print(len(well_code_list))

rmse_sum = 0
mape_sum = 0
n_rmses = 0
for well_code in well_code_list:
    well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
    print(well_name)
    well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])
    z_values = []      # List for z values
    temp_values = []   # List for temperature values
    first_elem = foft_csv.loc[foft_csv['Well']==well_code]['ELEM'].unique()[-1]
    elem_x, elem_y = dat.grid.block[first_elem].centre[0], dat.grid.block[first_elem].centre[1]
    # Iterate over blocks in same column
    for blk in dat.grid.blocklist[1:]:
        if blk.centre[0] == elem_x and blk.centre[1] == elem_y:
            z_values.append(blk.centre[2])  # Append z coord
            temp_values.append(inc[blk.name][1])    # Append temperature
    new_df_temp_z_model = pd.DataFrame(columns=['z', 'temp'])
    new_df_temp_z_model['temp'] = temp_values
    new_df_temp_z_model['z'] = z_values
    new_df_temp_z_model.sort_values(by='z', ascending=True, inplace=True)
    plt.figure(figsize=(7,15))
    plt.plot(new_df_temp_z_model['temp'], new_df_temp_z_model['z'], c='red', label='Modelled Temperature')
    # Plot field temperature data if available
    if well_long_name in wells_w_test_data:
        # Read excel file with test data
        f_water = os.path.join(test_data_TVD_dir, well_long_name+'TVD_log.csv')
        well_test_df = pd.read_csv(f_water)
        # Delete nan depth rows
        well_test_df = well_test_df[well_test_df['TVD (m)'].notna()]
        plt.scatter(well_test_df['Well temperature (°C)'], -well_test_df['TVD (m)'], c='red', label='Measured Temperature')
        
        # Create new dataframe to interpoalate well depth values to modelled temperature
        # and make it possible to calculate RMS Error
        df_z = pd.concat([new_df_temp_z_model['z'],  -well_test_df['TVD (m)']], axis=0, ignore_index=True)
        df_temp = pd.concat([new_df_temp_z_model['temp'],  pd.DataFrame([np.nan]*len(well_test_df['TVD (m)']))], axis=0, ignore_index=True)
        df_ztemp_interpolate = pd.concat([df_z, df_temp], axis=1)
        df_ztemp_interpolate.columns = ['z', 'temp']
        df_ztemp_interpolate.sort_values(by='z', ascending=False, inplace=True)
        df_ztemp_interpolate.drop_duplicates()
        df_ztemp_interpolate.reset_index()
        df_ztemp_interpolate.interpolate(method='linear', inplace=True)
        
        # New dataframe with modelled T and measured T to compare with RMS Error
        df_rms_calc = df_ztemp_interpolate[df_ztemp_interpolate['z'].isin(-well_test_df['TVD (m)'])]
        # RMS Error and sccatter index
        # mse_man = np.square(np.subtract(well_test_df['Well temperature (°C)'][5:-3], df_rms_calc['temp'][5:-3])).mean()
        # rmse_man = np.sqrt(mse_man)
        rmse = np.sqrt(mean_squared_error(well_test_df['Well temperature (°C)'][5:-3], df_rms_calc['temp'][5:-3]))
        # Scatter Index
        scat_idx = rmse*100/np.mean(well_test_df['Well temperature (°C)'][5:-3])
        # Mean Absolute Percentage Error
        mape = mean_absolute_percentage_error(well_test_df['Well temperature (°C)'][5:-3], df_rms_calc['temp'][5:-3])*100
        
        rmse_sum += rmse
        mape_sum += mape
        n_rmses += 1
        # Scatter interpolated modelled temperature from well depths
        plt.scatter(df_rms_calc['temp'][5:-3], df_rms_calc['z'][5:-3], c='darksalmon', marker='x', label='Interp. Well Model Temperature')

        plt.text(51, -300, f'RMSE: {rmse:.3f} \nSI: {scat_idx:.3f} %\nMAPE: {mape:.3f} %')
        print(well_long_name+' -- RMSE: '+f'{rmse:.3f}, SI: {scat_idx:.3f} %, MAPE: {mape:.3f} %')
        
    plt.grid(color='grey', linestyle = '--', linewidth = 0.5)
    plt.xlim((24,65))
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Depth (m)')
    plt.title(well_long_name)
    plt.legend()
    plt.savefig(ns_calibration_dir+well_name + ' ' + well_code+'Temp.png', transparent=False)
    plt.close()

print(f'-----NATURAL STATE CALIBRATION-----\nAverage RMSE: {rmse_sum/n_rmses:.3f}\nAverage MAPE: {mape_sum/n_rmses:.3f} %') 


# for well_code in well_code_list:
#     well_name = str(df_well_names_codes.loc[df_well_names_codes['Well_code']==well_code]['Well_name'].values[0])
#     well_long_name = str(df_long_names.loc[df_long_names['Well short name']==well_name]['Well name'].values[0])
#     z_values = []      # List for z values
#     temp_values = []   # List for temperature values
#     elem_list = foft_csv.loc[foft_csv['Well']==well_code]['ELEM'].unique()
#     for elem in elem_list:
#         try:
#             temp_values.append(inc[elem][1])    # Append temperature
#             block_name = dat.grid.block[elem]
#             z_values.append(block_name.centre[2])
#         except:
#             elem = elem[:3] + '0' + elem[4:]
#             temp_values.append(inc[elem][1])    # Append temperature
#             block_name = dat.grid.block[elem]
#             z_values.append(block_name.centre[2])
#     new_df_temp_z_model = pd.DataFrame(columns=['z', 'temp'])
#     new_df_temp_z_model['temp'] = temp_values
#     new_df_temp_z_model['z'] = z_values
#     new_df_temp_z_model.sort_values(by='z', ascending=True, inplace=True)
#     plt.figure(figsize=(7,15))
#     plt.plot(new_df_temp_z_model['temp'], new_df_temp_z_model['z'], c='red', label='Modelled Temperature')
#     # Plot field temperature data if available
#     if well_long_name in wells_w_test_data:
#         # Read excel file with test data
#         f_water = os.path.join(test_data_TVD_dir, well_long_name+'TVD_log.csv')
#         well_test_df = pd.read_csv(f_water)
#         plt.scatter(well_test_df['Well temperature (°C)'], -well_test_df['TVD (m)'], c='red', label='Measured Temperature')

#     plt.grid(color='grey', linestyle = '--', linewidth = 0.5)
#     plt.xlim((30,60))
#     plt.title(well_long_name)
#     plt.savefig(ns_calibration_dir+well_name + ' ' + well_code+'Temp.png')
#     plt.legend()
#     plt.close()