from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
from xgboost import DMatrix
#from xgboost import train
from xgboost import plot_importance

import os
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

def prepare_data_riegel(data,targetparam,system_trigger,systemwahl,featureset):
    
#### Auflistung der Feature. Index zu Name
    #for i in range(len(data.columns)):
    #    print(i, ":", data.columns[i])
    #for i in range(200):
       # print(i, ":", data.columns[i])
    #    print(data.columns[i],"= [",i,"]")
    #print(len(data.columns)-1,":",data.columns[-1])


#### Lesbarmachen der Featurenamen-Indexe. Wichtig für die Auswahl der Features


    binder_HE=[0,1,2,3]
    sand_HE=[4,5,6,7]
    systembuchstabe_HE=list(range(8,19))
    System_Buchstabe=[19]
    Charge_Nummer=[20]
    Riegel_Nummer=[21]
    mittlerer_Korndurchmesser=[22]
    Bindermenge=[23]
    Festigkeit_MPa_HA=[24]
    Mittelwert_Schlagzahl=[25]
    Stabw_Schlagzahl=[26]
    Anzahl_Messwerte_Schlagzahl=[27]
    Entkernrate_masse_pro_Schlag=[28]
    Stabw_Entkernrate=[29]
    Biegefestigkeit_roh_MPa=[30]
    Stabw_Biegefestigkeit_roh_MPa=[31]
    Anzahl_Messwerte_Biegefestigkeit_roh_MPa=[32]
    Druckfestigkeit_roh_MPa=[33]
    Stabw_Druckfestigkeit_roh_MPa=[34]
    Anzahl_Messwerte_Druckfestigkeit_roh_MPa=[35]
    Winkel_phi_Drucker_Prager=[36]
    Stabw_Winkel_phi_Drucker_Prager=[37]
    Kohaesion_c1=[38]
    Stabw_Kohaesion_c1=[39]
    Biegefestigkeit_400C_MPa=[40]
    Stabw_Biegefestigkeit_400C_MPa=[41]
    Anzahl_Messwerte_Biegefestigkeit_400C_MPa=[42]
    Druckfestigkeit_400C_MPa=[43]
    Stabw_Druckfestigkeit_400C_MPa=[44]
    Anzahl_Messwerte_Druckfestigkeit_400C_MPa=[45]
    Winkel_phi_Drucker_Prager_400C=[46]
    Stabw_Winkel_phi_Drucker_Prager_400C=[47]
    Kohaesion_c1_400C=[48]
    Stabw_Kohaesion_c1_400C=[49]
    Biegefestigkeit_roh_MPa_600C=[50]
    Stabw_Biegefestigkeit_roh_MPa_600C=[51]
    Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C=[52]
    Druckfestigkeit_roh_MPa_600C=[53]
    Stabw_Druckfestigkeit_roh_MPa_600C=[54]
    Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C=[55]
    Biegefestigkeit_roh_MPa_750C=[56]
    Stabw_Biegefestigkeit_roh_MPa_750C=[57]
    Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C=[58]
    Druckfestigkeit_roh_MPa_750C=[59]
    Stabw_Druckfestigkeit_roh_MPa_750C=[60]
    Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C=[61]
    Winkel_phi_Drucker_Prager_750C=[62]
    Stabw_Winkel_phi_Drucker_Prager_750C=[63]
    Kohaesion_c1_750C=[64]
    Stabw_Kohaesion_c1_750C=[65]
    Biegefestigkeit_eingegossen_MPa=[66]
    Stabw_Biegefestigkeit_eingegossen_MPa=[67]
    Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa=[68]
    Druckfestigkeit_eingegossen_MPa=[69]
    Stabw_Druckfestigkeit_eingegossen_MPa=[70]
    Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa=[71]
    Winkel_phi_Drucker_Prager_eingegossen=[72]
    Stabw_Winkel_phi_Drucker_Prager_eingegossen=[73]
    Kohaesion_c1_eingegossen=[74]
    Stabw_Winkel_Kohaesion_c1_eingegossen=[75]
    Spannung_DMS_quer_Mitte_MPa=[76]
    Stabw_Spannung_DMS_quer_Mitte_MPa=[77]
    Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa=[78]
    Spannung_DMS_quer_Anguss_MPa=[79]
    Stabw_DMS_quer_Anguss_MPa=[80]
    Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa=[81]
    Spannung_DMS_laengs_Mitte_MPa=[82]
    Stabw_DMS_laengs_Mitte_MPa=[83]
    Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa=[84]
    Siebdurchgang_HA_Prozent=[85]
    Stabw_Siebdurchgang_HA_Prozent=[86]
    Anzahl_Messwerte_Siebdurchgang_HA_Prozent=[87]
    Restfestigkeit_HA_Prozent=[88]
    Stabw_Restfestigkeit_HA_Prozent=[89]
    Anzahl_Messwerte_Restfestigkeit_HA_Prozent=[90]
    Restfestigkeit_HA_MPa=[91]
    Stabw_Restfestigkeit_HA_MPa=[92]
    Anzahl_Messwerte_Restfestigkeit_HA_MPa=[93]
    Abfall_Biegefestigkeit_Prozent=[94]
    B_n_intervalls_needed_to_decore=[95]
    B_start_mass_abs=[96]
    B_start_distance_abs=[97]
    B_sandmass=[98]
    B_sandmass_rel_totalmass=[99]
    B_sandmass_rel_metalmass=[100]
    B_average_mass_per_I_abs=[101]
    B_average_distance_per_I_abs=[102]
    B_average_mass_per_I_rel_totalmassdelta=[103]
    B_average_distance_per_I_rel_totaldistance=[104]
    B_n_hammerblows=[105]
    B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow=[106]
    B_sum_of_max_positive_acceleration_of_each_hammerblow=[107]
    B_sum_of_max_negative_acceleration_of_each_hammerblow=[108]
    B_sum_max_positive_acceleration_of_each_intervall=[109]
    B_sum_max_negative_acceleration_of_each_intervall=[110]
    B_mean_of_max_positive_acceleration_of_each_hammerblow=[111]
    B_mean_of_max_negative_acceleration_of_each_hammerblow=[112]
    hammer_mfcc_feature1=[113,118,123,128,133]
    hammer_mfcc_feature2=[114,119,124,129,134]
    hammer_mfcc_feature3=[115,120,125,130,135]
    hammer_mfcc_feature4=[116,121,126,131,136]
    hammer_mfcc_feature5=[117,122,127,132,137]
    FFT_Werte=list(range(138,4938))
  
####
   # Hier die Auswahl der Features vornehmen. 
   # Auskommentiert = Ausgewählt
####
##droplist1: "Base1: Maximale Featurezahl mit Systemen ABDL ohne Entkernfortschritt, mit FFT,mfccc,Signal". 
    droplist1=([]
    #### Allgemein ####
        #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    +systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    +binder_HE
    +sand_HE
    +Bindermenge
    #+Siebdurchgang_HA_Prozent
    #+mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    +FFT_Werte

    #### mfcc features ####
    +hammer_mfcc_feature1
    +hammer_mfcc_feature2
    +hammer_mfcc_feature3
    +hammer_mfcc_feature4
    +hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    +B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_negative_acceleration_of_each_hammerblow
    +B_sum_max_positive_acceleration_of_each_intervall
    +B_sum_max_negative_acceleration_of_each_intervall
    +B_mean_of_max_positive_acceleration_of_each_hammerblow
    +B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    #+Biegefestigkeit_400C_MPa
    #+Druckfestigkeit_400C_MPa
    #+Winkel_phi_Drucker_Prager_400C
    #+Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    #+Biegefestigkeit_eingegossen_MPa
    #+Druckfestigkeit_eingegossen_MPa
    #+Winkel_phi_Drucker_Prager_eingegossen
    #+Kohaesion_c1_eingegossen
    #+Restfestigkeit_HA_Prozent
    #+Restfestigkeit_HA_MPa
    #+Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    #+Spannung_DMS_quer_Mitte_MPa
    #+Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
##droplist2: "Base2: Maximale Zahl an Daten, Alle Systeme, ohne Entkernfortschritt", mit FFT,mfccc,Signal
    droplist2=([]
    #### Allgemein ####
    #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    +systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    +binder_HE
    +sand_HE
    +Bindermenge
    #+Siebdurchgang_HA_Prozent
    +mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    +FFT_Werte

    #### mfcc features ####
    +hammer_mfcc_feature1
    +hammer_mfcc_feature2
    +hammer_mfcc_feature3
    +hammer_mfcc_feature4
    +hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    +B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_negative_acceleration_of_each_hammerblow
    +B_sum_max_positive_acceleration_of_each_intervall
    +B_sum_max_negative_acceleration_of_each_intervall
    +B_mean_of_max_positive_acceleration_of_each_hammerblow
    +B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    +Biegefestigkeit_400C_MPa
    +Druckfestigkeit_400C_MPa
    +Winkel_phi_Drucker_Prager_400C
    +Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    +Biegefestigkeit_eingegossen_MPa
    +Druckfestigkeit_eingegossen_MPa
    +Winkel_phi_Drucker_Prager_eingegossen
    +Kohaesion_c1_eingegossen
    +Restfestigkeit_HA_Prozent
    +Restfestigkeit_HA_MPa
    +Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    +Spannung_DMS_quer_Mitte_MPa
    +Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
##droplist3: "Keine Buchstaben HE, Maximale Featurezahl mit Systemen ABDL ohne Entkernfortschritt, mit FFT,mfccc,Signal". 
    droplist3=([]
    #### Allgemein ####
    #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    #+systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    #+binder_HE
    #+sand_HE
    #+Bindermenge
    #+Siebdurchgang_HA_Prozent
    #+mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    +FFT_Werte

    #### mfcc features ####
    +hammer_mfcc_feature1
    +hammer_mfcc_feature2
    +hammer_mfcc_feature3
    +hammer_mfcc_feature4
    +hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    +B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_negative_acceleration_of_each_hammerblow
    +B_sum_max_positive_acceleration_of_each_intervall
    +B_sum_max_negative_acceleration_of_each_intervall
    +B_mean_of_max_positive_acceleration_of_each_hammerblow
    +B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    #+Biegefestigkeit_400C_MPa
    #+Druckfestigkeit_400C_MPa
    #+Winkel_phi_Drucker_Prager_400C
    #+Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    #+Biegefestigkeit_eingegossen_MPa
    #+Druckfestigkeit_eingegossen_MPa
    #+Winkel_phi_Drucker_Prager_eingegossen
    #+Kohaesion_c1_eingegossen
    #+Restfestigkeit_HA_Prozent
    #+Restfestigkeit_HA_MPa
    #+Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    #+Spannung_DMS_quer_Mitte_MPa
    #+Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
##droplist4: "Keine Buchstaben HE, Maximale Zahl an Daten, Alle Systeme, ohne Entkernfortschritt", mit FFT,mfccc,Signal
    droplist4=([]
    #### Allgemein ####
    #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    #+systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    #+binder_HE
    #+sand_HE
    #+Bindermenge
    #+Siebdurchgang_HA_Prozent
    +mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    +FFT_Werte

    #### mfcc features ####
    +hammer_mfcc_feature1
    +hammer_mfcc_feature2
    +hammer_mfcc_feature3
    +hammer_mfcc_feature4
    +hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    +B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_negative_acceleration_of_each_hammerblow
    +B_sum_max_positive_acceleration_of_each_intervall
    +B_sum_max_negative_acceleration_of_each_intervall
    +B_mean_of_max_positive_acceleration_of_each_hammerblow
    +B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    +Biegefestigkeit_400C_MPa
    +Druckfestigkeit_400C_MPa
    +Winkel_phi_Drucker_Prager_400C
    +Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    +Biegefestigkeit_eingegossen_MPa
    +Druckfestigkeit_eingegossen_MPa
    +Winkel_phi_Drucker_Prager_eingegossen
    +Kohaesion_c1_eingegossen
    +Restfestigkeit_HA_Prozent
    +Restfestigkeit_HA_MPa
    +Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    +Spannung_DMS_quer_Mitte_MPa
    +Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
##droplist5: "Maximale Featurezahl mit Systemen ABDL ohne Entkernfortschritt", ohne FFT,mfccc,Signal 
    droplist5=([]
    #### Allgemein ####
        #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    +systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    +binder_HE
    +sand_HE
    +Bindermenge
    #+Siebdurchgang_HA_Prozent
    #+mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    #+FFT_Werte

    #### mfcc features ####
    #+hammer_mfcc_feature1
    #+hammer_mfcc_feature2
    #+hammer_mfcc_feature3
    #+hammer_mfcc_feature4
    #+hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    #+B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    #+B_sum_of_max_positive_acceleration_of_each_hammerblow
    #+B_sum_of_max_negative_acceleration_of_each_hammerblow
    #+B_sum_max_positive_acceleration_of_each_intervall
    #+B_sum_max_negative_acceleration_of_each_intervall
    #+B_mean_of_max_positive_acceleration_of_each_hammerblow
    #+B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    #+Biegefestigkeit_400C_MPa
    #+Druckfestigkeit_400C_MPa
    #+Winkel_phi_Drucker_Prager_400C
    #+Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    #+Biegefestigkeit_eingegossen_MPa
    #+Druckfestigkeit_eingegossen_MPa
    #+Winkel_phi_Drucker_Prager_eingegossen
    #+Kohaesion_c1_eingegossen
    #+Restfestigkeit_HA_Prozent
    #+Restfestigkeit_HA_MPa
    #+Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    #+Spannung_DMS_quer_Mitte_MPa
    #+Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
##droplist6: "Maximale Zahl an Daten, Alle Systeme, ohne Entkernfortschritt",ohne FFT,mfccc,Signal.
    droplist6=([]
    #### Allgemein ####
    #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    +systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    +binder_HE
    +sand_HE
    +Bindermenge
    #+Siebdurchgang_HA_Prozent
    +mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    #+FFT_Werte

    #### mfcc features ####
    #+hammer_mfcc_feature1
    #+hammer_mfcc_feature2
    #+hammer_mfcc_feature3
    #+hammer_mfcc_feature4
    #+hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    #+B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    #+B_sum_of_max_positive_acceleration_of_each_hammerblow
    #+B_sum_of_max_negative_acceleration_of_each_hammerblow
    #+B_sum_max_positive_acceleration_of_each_intervall
    #+B_sum_max_negative_acceleration_of_each_intervall
    #+B_mean_of_max_positive_acceleration_of_each_hammerblow
    #+B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    +Biegefestigkeit_400C_MPa
    +Druckfestigkeit_400C_MPa
    +Winkel_phi_Drucker_Prager_400C
    +Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    +Biegefestigkeit_eingegossen_MPa
    +Druckfestigkeit_eingegossen_MPa
    +Winkel_phi_Drucker_Prager_eingegossen
    +Kohaesion_c1_eingegossen
    +Restfestigkeit_HA_Prozent
    +Restfestigkeit_HA_MPa
    +Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    +Spannung_DMS_quer_Mitte_MPa
    +Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
##droplist7: "FFT: Maximale Featurezahl mit Systemen ABDL ohne Entkernfortschritt", mit FFT, ohne mfccc,Signal 
    droplist7=([]
    #### Allgemein ####
        #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    +systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    +binder_HE
    +sand_HE
    +Bindermenge
    #+Siebdurchgang_HA_Prozent
    #+mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    #+FFT_Werte

    #### mfcc features ####
    +hammer_mfcc_feature1
    +hammer_mfcc_feature2
    +hammer_mfcc_feature3
    +hammer_mfcc_feature4
    +hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    +B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_negative_acceleration_of_each_hammerblow
    +B_sum_max_positive_acceleration_of_each_intervall
    +B_sum_max_negative_acceleration_of_each_intervall
    +B_mean_of_max_positive_acceleration_of_each_hammerblow
    +B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    #+Biegefestigkeit_400C_MPa
    #+Druckfestigkeit_400C_MPa
    #+Winkel_phi_Drucker_Prager_400C
    #+Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    #+Biegefestigkeit_eingegossen_MPa
    #+Druckfestigkeit_eingegossen_MPa
    #+Winkel_phi_Drucker_Prager_eingegossen
    #+Kohaesion_c1_eingegossen
    #+Restfestigkeit_HA_Prozent
    #+Restfestigkeit_HA_MPa
    #+Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    #+Spannung_DMS_quer_Mitte_MPa
    #+Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
##droplist8: "FFT: Maximale Zahl an Daten, Alle Systeme, ohne Entkernfortschritt", mit FFT, ohne mfccc,Signal.
    droplist8=([]
    #### Allgemein ####
    #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    +systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    +binder_HE
    +sand_HE
    +Bindermenge
    #+Siebdurchgang_HA_Prozent
    +mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    #+FFT_Werte

    #### mfcc features ####
    +hammer_mfcc_feature1
    +hammer_mfcc_feature2
    +hammer_mfcc_feature3
    +hammer_mfcc_feature4
    +hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    +B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_negative_acceleration_of_each_hammerblow
    +B_sum_max_positive_acceleration_of_each_intervall
    +B_sum_max_negative_acceleration_of_each_intervall
    +B_mean_of_max_positive_acceleration_of_each_hammerblow
    +B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    +Biegefestigkeit_400C_MPa
    +Druckfestigkeit_400C_MPa
    +Winkel_phi_Drucker_Prager_400C
    +Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    +Biegefestigkeit_eingegossen_MPa
    +Druckfestigkeit_eingegossen_MPa
    +Winkel_phi_Drucker_Prager_eingegossen
    +Kohaesion_c1_eingegossen
    +Restfestigkeit_HA_Prozent
    +Restfestigkeit_HA_MPa
    +Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    +Spannung_DMS_quer_Mitte_MPa
    +Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
##droplist9: "mfcc: Maximale Featurezahl mit Systemen ABDL ohne Entkernfortschritt", mit mfcc, ohne FFT mfccc,Signal 
    droplist9=([]
    #### Allgemein ####
        #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    +systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    +binder_HE
    +sand_HE
    +Bindermenge
    #+Siebdurchgang_HA_Prozent
    #+mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    +FFT_Werte

    #### mfcc features ####
    #+hammer_mfcc_feature1
    #+hammer_mfcc_feature2
    #+hammer_mfcc_feature3
    #+hammer_mfcc_feature4
    #+hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    +B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_negative_acceleration_of_each_hammerblow
    +B_sum_max_positive_acceleration_of_each_intervall
    +B_sum_max_negative_acceleration_of_each_intervall
    +B_mean_of_max_positive_acceleration_of_each_hammerblow
    +B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    #+Biegefestigkeit_400C_MPa
    #+Druckfestigkeit_400C_MPa
    #+Winkel_phi_Drucker_Prager_400C
    #+Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    #+Biegefestigkeit_eingegossen_MPa
    #+Druckfestigkeit_eingegossen_MPa
    #+Winkel_phi_Drucker_Prager_eingegossen
    #+Kohaesion_c1_eingegossen
    #+Restfestigkeit_HA_Prozent
    #+Restfestigkeit_HA_MPa
    #+Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    #+Spannung_DMS_quer_Mitte_MPa
    #+Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
##droplist10: "mfcc: Maximale Zahl an Daten, Alle Systeme, ohne Entkernfortschritt", mit mfcc, ohne FFT mfccc,Signal.
    droplist10=([]
    #### Allgemein ####
    #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    +systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    +binder_HE
    +sand_HE
    +Bindermenge
    #+Siebdurchgang_HA_Prozent
    +mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    +FFT_Werte

    #### mfcc features ####
    #+hammer_mfcc_feature1
    #+hammer_mfcc_feature2
    #+hammer_mfcc_feature3
    #+hammer_mfcc_feature4
    #+hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    +B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_positive_acceleration_of_each_hammerblow
    +B_sum_of_max_negative_acceleration_of_each_hammerblow
    +B_sum_max_positive_acceleration_of_each_intervall
    +B_sum_max_negative_acceleration_of_each_intervall
    +B_mean_of_max_positive_acceleration_of_each_hammerblow
    +B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    +Biegefestigkeit_400C_MPa
    +Druckfestigkeit_400C_MPa
    +Winkel_phi_Drucker_Prager_400C
    +Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    +Biegefestigkeit_eingegossen_MPa
    +Druckfestigkeit_eingegossen_MPa
    +Winkel_phi_Drucker_Prager_eingegossen
    +Kohaesion_c1_eingegossen
    +Restfestigkeit_HA_Prozent
    +Restfestigkeit_HA_MPa
    +Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    +Spannung_DMS_quer_Mitte_MPa
    +Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
##droplist11: "Signal: Maximale Featurezahl mit Systemen ABDL ohne Entkernfortschritt", mit Signal, ohne FFT, mfccc 
    droplist11=([]
    #### Allgemein ####
        #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    +systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    +binder_HE
    +sand_HE
    +Bindermenge
    #+Siebdurchgang_HA_Prozent
    #+mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    +FFT_Werte

    #### mfcc features ####
    +hammer_mfcc_feature1
    +hammer_mfcc_feature2
    +hammer_mfcc_feature3
    +hammer_mfcc_feature4
    +hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    #+B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    #+B_sum_of_max_positive_acceleration_of_each_hammerblow
    #+B_sum_of_max_negative_acceleration_of_each_hammerblow
    #+B_sum_max_positive_acceleration_of_each_intervall
    #+B_sum_max_negative_acceleration_of_each_intervall
    #+B_mean_of_max_positive_acceleration_of_each_hammerblow
    #+B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    #+Biegefestigkeit_400C_MPa
    #+Druckfestigkeit_400C_MPa
    #+Winkel_phi_Drucker_Prager_400C
    #+Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    #+Biegefestigkeit_eingegossen_MPa
    #+Druckfestigkeit_eingegossen_MPa
    #+Winkel_phi_Drucker_Prager_eingegossen
    #+Kohaesion_c1_eingegossen
    #+Restfestigkeit_HA_Prozent
    #+Restfestigkeit_HA_MPa
    #+Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    #+Spannung_DMS_quer_Mitte_MPa
    #+Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
##droplist12: "Signal: Maximale Zahl an Daten, Alle Systeme, ohne Entkernfortschritt", mit Signal, ohne FFT, mfccc.
    droplist12=([]
    #### Allgemein ####
    #+System_Buchstabe #immer auskommentieren, wird für Aufbereitung benötigt und später gelöscht

    #### Zusammensetzung ####
    +systembuchstabe_HE                                 #enthält die restliche Zusammensetzung in Summe. Für Robustheit vielleicht lieber die Einzelwerte als die Summe? 
    +binder_HE
    +sand_HE
    +Bindermenge
    #+Siebdurchgang_HA_Prozent
    +mittlerer_Korndurchmesser

    ### Riegeldaten
    +B_n_intervalls_needed_to_decore #enthält Ergebnis
    #+B_start_mass_abs
    #+B_start_distance_abs
    #+B_sandmass
    #+B_sandmass_rel_totalmass
    #+B_sandmass_rel_metalmass
    +B_average_mass_per_I_abs #enthält Ergebnis
    +B_average_distance_per_I_abs #enthält Ergebnis
    +B_average_mass_per_I_rel_totalmassdelta #enthält Ergebnis
    +B_average_distance_per_I_rel_totaldistance #enthält Ergebnis

    #### FFT Werte 19200 ####
    +FFT_Werte

    #### mfcc features ####
    +hammer_mfcc_feature1
    +hammer_mfcc_feature2
    +hammer_mfcc_feature3
    +hammer_mfcc_feature4
    +hammer_mfcc_feature5

    #### Signal features (Maximum, Minimum) ####
    #+B_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow
    #+B_sum_of_max_positive_acceleration_of_each_hammerblow
    #+B_sum_of_max_negative_acceleration_of_each_hammerblow
    #+B_sum_max_positive_acceleration_of_each_intervall
    #+B_sum_max_negative_acceleration_of_each_intervall
    #+B_mean_of_max_positive_acceleration_of_each_hammerblow
    #+B_mean_of_max_negative_acceleration_of_each_hammerblow

    #### Grund-Festigkeiten ####
    +Festigkeit_MPa_HA #nicht verwenden
    #+Biegefestigkeit_roh_MPa
    #+Druckfestigkeit_roh_MPa
    #+Winkel_phi_Drucker_Prager
    #+Kohaesion_c1

    #### Temperatur Festigkeiten ####

    +Biegefestigkeit_400C_MPa
    +Druckfestigkeit_400C_MPa
    +Winkel_phi_Drucker_Prager_400C
    +Kohaesion_c1_400C
    +Biegefestigkeit_roh_MPa_600C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_600C #zu wenig Daten
    +Biegefestigkeit_roh_MPa_750C #zu wenig Daten
    +Druckfestigkeit_roh_MPa_750C #zu wenig Daten
    +Winkel_phi_Drucker_Prager_750C #zu wenig Daten
    +Kohaesion_c1_750C #zu wenig Daten

    #### Festigkeiten nach Abguss ####
    +Biegefestigkeit_eingegossen_MPa
    +Druckfestigkeit_eingegossen_MPa
    +Winkel_phi_Drucker_Prager_eingegossen
    +Kohaesion_c1_eingegossen
    +Restfestigkeit_HA_Prozent
    +Restfestigkeit_HA_MPa
    +Abfall_Biegefestigkeit_Prozent

    #### Spannungen ####
    +Spannung_DMS_quer_Mitte_MPa
    +Spannung_DMS_quer_Anguss_MPa
    +Spannung_DMS_laengs_Mitte_MPa


    ###### sonstige Werte ###### 
    +Charge_Nummer
    +Riegel_Nummer   
    +Mittelwert_Schlagzahl #enthält Ergebnis
    +Stabw_Schlagzahl  #enthält Ergebnis so
    +Anzahl_Messwerte_Schlagzahl #enthält Ergebnis so
    +Entkernrate_masse_pro_Schlag #enthält Ergebnis
    +Stabw_Entkernrate #enthält Ergebnis so
    +Stabw_Biegefestigkeit_roh_MPa
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa
    +Stabw_Druckfestigkeit_roh_MPa
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa
    +Stabw_Winkel_phi_Drucker_Prager
    +Stabw_Kohaesion_c1
    +Stabw_Biegefestigkeit_400C_MPa
    +Anzahl_Messwerte_Biegefestigkeit_400C_MPa
    +Stabw_Druckfestigkeit_400C_MPa
    +Anzahl_Messwerte_Druckfestigkeit_400C_MPa
    +Stabw_Winkel_phi_Drucker_Prager_400C
    +Stabw_Kohaesion_c1_400C 
    +Stabw_Biegefestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_600C
    +Stabw_Druckfestigkeit_roh_MPa_600C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_600C   
    +Stabw_Biegefestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Biegefestigkeit_roh_MPa_750C
    +Stabw_Druckfestigkeit_roh_MPa_750C
    +Anzahl_Messwerte_Druckfestigkeit_roh_MPa_750C    
    +Stabw_Winkel_phi_Drucker_Prager_750C   
    +Stabw_Kohaesion_c1_750C 
    +Stabw_Biegefestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Biegefestigkeit_eingegossen_MPa   
    +Stabw_Druckfestigkeit_eingegossen_MPa
    +Anzahl_Messwerte_Druckfestigkeit_eingegossen_MPa  
    +Stabw_Winkel_phi_Drucker_Prager_eingegossen  
    +Stabw_Winkel_Kohaesion_c1_eingegossen  
    +Stabw_Spannung_DMS_quer_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Mitte_MPa
    +Stabw_DMS_quer_Anguss_MPa
    +Anzahl_Messwerte_Spannung_DMS_quer_Anguss_MPa  
    +Stabw_DMS_laengs_Mitte_MPa
    +Anzahl_Messwerte_Spannung_DMS_laengs_Mitte_MPa 
    +Stabw_Siebdurchgang_HA_Prozent
    +Anzahl_Messwerte_Siebdurchgang_HA_Prozent
    +Stabw_Restfestigkeit_HA_Prozent
    +Anzahl_Messwerte_Restfestigkeit_HA_Prozent
    +Stabw_Restfestigkeit_HA_MPa
    +Anzahl_Messwerte_Restfestigkeit_HA_MPa
    )
###### Ende Featureauswahl
    droplist=[[],droplist1,droplist2,droplist3,droplist4,droplist5,droplist6,droplist7,droplist8,droplist9,droplist10,droplist11,droplist12]
    droplist=droplist[featureset]

#### Datenauswahl durchführen
    modeldata=data.drop(data.columns[droplist],axis=1)


#### Löschen aller Zeilen, die NAN enthalten
    
    modeldata.dropna(axis='rows',inplace=True)

### Spalten löschen, die nur 0 enthalten (hotencoded)

    modeldata = modeldata.loc[:, (modeldata != 0.0).any(axis=0)]

#### Ausreißer löschen

    modeldata.drop(modeldata[modeldata[targetparam[0]] > targetparam[1]].index, inplace=True)
    modeldata.drop(modeldata[modeldata[targetparam[0]] < targetparam[2]].index, inplace=True)

#### Auslesen der maximalen und minimalen Zielwertgröße für die spätere Umrechnung des skalierten Fehlers zurück in absolute Hammerschläge
    targetparam[3]=modeldata[targetparam[0]].max()
    targetparam[4]=modeldata[targetparam[0]].min()
    targetparam[5]=modeldata[targetparam[0]].mean()
    targetparam[6]=modeldata[targetparam[0]].median()

### Filtern nach Systemen, falls aktiviert
    print(modeldata.value_counts("system"))
    if system_trigger==True:
        modeldata_system=pd.DataFrame()
        modeldata_system=modeldata[modeldata['system'].isin(systemwahl)].copy()
        modeldata_system.drop('system',axis=1,inplace=True)

        modeldata=modeldata.drop(modeldata[modeldata['system'].isin(systemwahl)].index)
        modeldata.drop('system',axis=1,inplace=True)

        X_valsystem=modeldata_system.drop(targetparam[0],axis=1)
        y_valsystem=modeldata_system[targetparam[0]]
    else:
        modeldata.drop('system',axis=1,inplace=True)
        X_valsystem=0
        y_valsystem=0
    y=modeldata[targetparam[0]]
    modeldata.drop(targetparam[0],axis=1,inplace=True)
    

#### Skalieren der Features. Spaltenweise auf Werte zwischen 0 und 1 skalieren.
    #System wird benötigt für spätere Optionen, daher wird für das Skaling die Spalte auf 0 gesetzt und nach dem Skalieren wieder mit den Originalwerten gefüllt
    #scaler = preprocessing.RobustScaler(with_centering=True,with_scaling=True,quantile_range=(5,95),copy=True)


#### Falls Zielgroesse nicht skaliert werden sollen
    #modeldata['Intervall-Fortschritt Gewicht absolut 03']=data['Intervall-Fortschritt Gewicht absolut 03'].reset_index(drop=True)

#### Ausgabe der verarbeiteten Daten
    return modeldata,y,X_valsystem,y_valsystem,targetparam
