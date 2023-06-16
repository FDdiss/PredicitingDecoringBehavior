###Versionen der Datensets
# v7: Variable Signallänge, resampling fft
# v8: Reduktion Signallänge auf 9600 Datenpunkte. FFT features auf 4800 reduziert (je 4 Hz). Dadurch sind alle FFT gleich (gleiche Signallänge, gleiche samplingrate) und somit real vergleichbar.
# v9: Hotencoding integriert, Featureanordnung geändert, Bereinigung der Featurenamen, Übersetzung der Features ins Englische, neue Feature hinzugefügt, Riegeldatenerstellung integriert
# v10: kleine Tippfehler ausgebessert, kleinere Fehler in den Roh-Daten bereinigt
# v11: übersprungen damit Code version = Datenset version 
# v12: Featurenamen verkürzt

import IPython.display as ipd
from joblib.logger import PrintTime
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import librosa
import statistics
from scipy import signal 
from scipy.io import wavfile as wav
import sys
import os
import matplotlib.pyplot as plt
import librosa.display

# Signaldaten einlesen
# Testsignal
string=r'G:\06 Studenten\Linke Moritz\Versuchsdaten Entkernen ett\Datenbasis_clean\2019-07-17 Aluminium E Charge 1/Riegel-E01-C1_00-20_38400Hz.MAT'

########## Einlese-Funktion für die Matlabdaten
# Channel 3 ist das Triggersignal
# Channel 4 ist das eigentliche Signal
# Übergeben wird der PFAD zum Signal
# Ausgabe: Signalarray, Triggerarray
# Signalarray mit dem Aufbau Signallänge*Signalzahl, wobei Signalzahl=Zahl Hammerschläge
#   Jede "Zeile" im Signalarray ist also ein Hammerschlagsignal
# Triggerarray ist eigentlich ein Vektor, dessen Länge der Anzahl Hammerschläge entspricht

def einlesen_mat_signal(string):

    mat = scipy.io.loadmat(string) #einlesen der Matlabdatei

    #Trigger auslesen

    triggersignal = mat["Channel_3_Data"] # Trigger aus Matlabdatei in Array speichern
    #plt.plot(triggersignal)
    #plt.show()
    triggerarray = np.zeros(20,dtype=int) # leeres Triggerarray für 20 Hammerschläge erstellen
    triggerlist=[]
    ### Triggersignal auslesen
    # Sorry für den Aufbau der Schleife, das war 2019, als ich Python noch nicht so konnte
    # Das Triggersignal wird mit Schwellwerten nach der steigenden Kante abgesucht
    # Es wird aber nur jedes zweite Signal akzeptiert, weil das Triggersignal pro Hammerschlag zwei steigende Kanten hat
    
    k = 0 #Positionszähler, da wir die Position der Kante brauchen
    lastsignal = triggersignal[0]
    for i in triggersignal[1:]: #einmal das gesamte Triggersignal abwandern
            if ((i > 2) and (lastsignal <= 2)): #ist der Datenpunkt größer 1.3 und der vorherige Datenpunkt kleiner 1.3, handelt es sich um eine steigende Kante
                if len(triggerlist)>0:
                    if int(k)-triggerlist[-1]>12000:
                        triggerlist.append(int(k))
                else:
                    triggerlist.append(int(k))
            k = k + 1 #Hochzählen des Positionszählers
            lastsignal = i #Speichern des aktuellen Signals in lastsignal für den nächsten Datenpunkt

    ### Channel 4 Hammerschlagsignal aufteilen in die einzelnen Hammerschläge
    # Wichtig für eine vergleichbare mfcc ist eine identische Signallänge für alle Hammerschläge!
    # Daher muss erst eine definierte Länge der Signale berechnet werden
    # Dafür wird der minimale Abstand zwischen den Triggersignalen verwendet
    # Bei 20 Hammerschlägen gibt es 19 Abstände

    #abstand = np.zeros(19) #Array zum Einspeichern der 19 Abstände
    #print("Laenge Triggerliste:", len(triggerlist))
    #print(triggerlist)
    abstandlist=[]
    lowcount=[0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(triggerlist)-1):
        abstandlist.append(triggerlist[i+1]-triggerlist[i]) #Berechnen der 19 Abstände

    # Länge der Signale wird auf min. Abstand zwischen Triggersignale minus 5 gesetzt

    signal_length = 9600 #Signallänge ist minimaler Abstand minus 5
    #print("SIGNALLAENGE:", signal_length)
    ### Zerschneiden des Signals in die einzelnen Hammerschläge

    signalarray=np.zeros((int(signal_length),len(triggerlist))) #Signalarray erstellen der korrekten Länge und Zahl Hammerschläge

    # Einspeichern der Signalstücke in das Signalarray
    maxlist=[]
    k=mat["Channel_4_Data"].shape[0]
    #plt.plot(mat["Channel_4_Data"])
    #plt.show()
    #print(k)
    #print(mat["Channel_4_Data"].shape)
    for i in range(len(triggerlist)): #Anzahl an Triggersignalen
        #von jeweiliger Triggerposition wird ein Signal der vorher definierten Signallänge ausgelesen
        m=int(triggerlist[i])+int(signal_length)
        #print("ENDPUNKT SIGNAL:", int(triggerlist[i])+int(signal_length))

        if m > k:
            #print("SIGNALARRAYGROESE_davor:",k)
            #print("SOLLGROESE:", m)
            #print(m-k)
            #print(mat["Channel_4_Data"])
            helperarray=np.concatenate((mat["Channel_4_Data"], np.zeros((m-k,1))),axis=0)
            #print("SIGNALARRAYGROESE_danach:",helperarray.size)
            signalarray[:,i]=helperarray[int(triggerlist[i]):m,0]
            #plt.plot(signalarray[:,i])
            #plt.show()
        else:
            signalarray[:,i]=mat["Channel_4_Data"][int(triggerlist[i]):m,0]
            #plt.plot(signalarray[:,i])
            #plt.show()
        max_sa=max(signalarray[:,i])
        if max_sa > 1000:
            #print(max(signalarray[:,i]))
            maxlist.append(max_sa)
        if max_sa < 5000:
            lowcount[0]=lowcount[0]+1
        if max_sa < 4000:
            lowcount[1]=lowcount[1]+1
        if max_sa < 3000:
            lowcount[2]=lowcount[2]+1
        if max_sa < 2000:
            lowcount[3]=lowcount[3]+1
        if max_sa < 1000:
            lowcount[4]=lowcount[4]+1
        if max_sa < 750:
            lowcount[5]=lowcount[5]+1
        if max_sa < 500:
            lowcount[6]=lowcount[6]+1
        if max_sa < 250:
            lowcount[7]=lowcount[7]+1
        if max_sa < 100:
            lowcount[8]=lowcount[8]+1
        if max_sa > 5000:
            lowcount[9]=lowcount[9]+1
        lowcount[10]=lowcount[10]+1
    #print(signalarray.shape) < 1000
    #print(~np.any(signalarray > 1000, axis=0))
    #print((~np.any(signalarray > 4000, axis=0)).shape)
    #if True in np.any(signalarray > 1000, axis=1) > 0:
    signalarray=np.delete(signalarray, ~np.any(signalarray > 1000, axis=0), axis=1)
    #print(signalarray.shape)
    print("Minimales MAXIMUM:", min(maxlist))
    print("Durchschnittliches MAXIMUM:", mean(maxlist))
    print("Maximales MAXIMUM:", max(maxlist))
    #Übergeben des Signalarrays und des Triggerarrays
    return signalarray, triggerlist, signal_length, lowcount

########## Signalauswertung, 
# welche einen Vektor der Länge [(num_hs+1)*num_mfcc_features*5 + num_hs*2 + 2], Standardlänge = 2142
# und eine Namensliste mit den Bezeichnungen der Features ausgibt
# num_hs ist die Zahl der Hammerschläge im Signal
# num_mfcc_features ist die Zahl der zu erstellenden mfcc_features pro Hammerschlag
# Genutzt wird eine Mel Frequency Cepstral Coefficients-Analyse (mfcc-Analyse)
# Außerdem wird für jeden Hammerschlag das Maximum und das Minimum berechnet

def generate_signalfeatures(matlab_path, num_mfcc_features=5, sr=38400):

    signalarray, triggerlist, signallength,lowcount = einlesen_mat_signal(matlab_path)
    num_hs=signalarray.shape[1]
    #print("NUMHS",num_hs)
    ### Erstellung der leeren Arrays, in welche die Features eingespeichert werden
    mfcc_mean = np.zeros((num_mfcc_features,num_hs+1)) #Ein array mit Shape Anzahl mfcc*(Anzahl Hammerschläge+1)
    mfcc_median = np.zeros((num_mfcc_features,num_hs+1))
    mfcc_stdev = np.zeros((num_mfcc_features,num_hs+1))
    mfcc_max = np.zeros((num_mfcc_features,num_hs+1))
    mfcc_min = np.zeros((num_mfcc_features,num_hs+1))
    absmaxarray=np.zeros((num_hs+1,1))
    absminarray=np.zeros((num_hs+1,1))

    length_fft = 4800 #38.400 hz sampling, Signallänge von 9600
    fft_hs = np.zeros((length_fft,num_hs+1))
    fft_mean=np.zeros((length_fft,1))
    max_abs_fft_discrepancy=0
    signallist=[]   

    ### Erste mfcc-Analyse zur Bestimmung der von der Analyse erzeugten Signalintervalle. 
    # Die Analyse teilt das übergebene Hammerschlag je nach Länge in sogeannte frames auf und deren Anzahl muss daher dynamisch berechnet werden
    hs_features=librosa.feature.mfcc(y=signalarray[:,0].T, sr=sr, n_mfcc = num_mfcc_features) #mfcc-Analyse des ersten Hammerschlags. Übergibt eine Matrix der Größe (num_features*num_frames)
    num_frames=len(hs_features[0,:]) #Auslesen der Anzahl der frames
    total_featurearray = np.zeros((num_mfcc_features,num_frames*num_hs)) #mit der Anzahl der frames kann nun ein leeres Array erzeugt werden, dass genau das gesamte Hammerschlagsignal umfasst

    ### Signalanalyse
    # Schleife geht die Hammerschläge durch
    # Für jeden Hammerschlag werden die mfcc berechnet
    # Über diese werden dann die fünf Auswertungen mean, median, stdev, max, min durchgeführt
    # Zusätzlich wird für jeden Hammerschlag aus dem Originalsignal das absolute Maximum und Minimum ausgelesen
    # Für alle Auswertungen wird außerdem noch ein globaler Wert=über alle Hammerschläge berechnet,
    # z.B. das absolute Signalmaximum aller Hammerschläge oder der Durchschnitt über eines Features über alle Hammerschläge

    # Start der Schleife
    for i in range(num_hs):
        signallist.append(signalarray[:,i])
        absmaxarray[i]=np.amax(signalarray[:,i]) #Auslesen des Maximums des Originalsignals des Hammerschlags
        absminarray[i]=np.amin(signalarray[:,i]) #Auslesen des Minimums des Originalsignals des Hammerschlags

        hs_features=librosa.feature.mfcc(y=signalarray[:,i].T, sr=sr, n_mfcc = num_mfcc_features) #mfcc Analyse des Hammerschlags
        
        #fft
        #fft_puffer=[0]*19200
        fft_signal=signalarray[:,i]
        #fft_signal=signal.resample(fft_signal,19200*2)
        fft_n=len(fft_signal)
        #print("Length signalarray: ",fft_n)
        fft_k=np.arange(1,fft_n+1)
        #print("Length fft_k: ",len(fft_k))
        fft_T=fft_n/sr
        #print("fft_T", fft_T)
        fft_freq_twosided=fft_k/fft_T
        #print("fft_freq_twosided",fft_freq_twosided)
        fft_freq=fft_freq_twosided[range(int(fft_n/2))]
        #print("fft Frequencys: ", fft_freq)
        #print("Length fft Frequencys: ", len(fft_freq))
        #print("Länge fft_werteliste", len(np.fft.fft(fft_signal)))
        #plt.plot(fft_freq_twosided,abs(np.fft.fft(fft_signal)[range(int(fft_n))]/fft_n),label='ganz')
        fft_values=abs(np.fft.fft(fft_signal)[range(int(fft_n/2))]/fft_n)
        #print("Minimum fft_values:", fft_values.min())
        #print(fft_puffer)
        #print("Length fft puffer: ", len(fft_puffer))
        #plt.plot(fft_freq, fft_values,label='original')
        #fft_puffer=abs(np.fft.fft(signalarray[:,i],19200)[0:]/fft_n)
        #fft_values=signal.resample(fft_values,int(fft_freq[-1]))
        #plt.plot(range(0,int(fft_freq[-1])), fft_values,label='resampled')
        #print("Minimum fft_values nach resampling:", fft_values.min())
        #plt.legend()
        #plt.show()
        
        
        #if len(fft_values) < 19200:
        #    fft_puffer[:len(fft_values)] = fft_values
        #else:
        #    fft_puffer=fft_values[0:19200]
        #plt.plot(range(0,19200), fft_puffer)
        #print(fft_puffer)
        #print("Length fft puffer: ", len(fft_puffer))
        
        fft_hs[:,i]=fft_values
        #if fft_freq[-1]/19200 < 0.9:
        #    print("frequency discrepancy")
        #    exit()
        #if fft_freq[-1]/19200 > 1.1:
        #    print("frequency discrepancy")
        #    exit()
        #max_abs_fft_discrepancy=max([max_abs_fft_discrepancy,abs(1-fft_freq[-1]/19200)])
        #print("max discrepancy: ", max_abs_fft_discrepancy)
        #quit()



        #num_frames=len(hs_features[0,:]) # wird durch die mfcc Analyse festgelegt und dann einfach ausgelesen
        
        ### Grafische Ausgabe der mfcc-Analse des Hammerschlags
        #fig, ax = plt.subplots()
        #img = librosa.display.specshow(hs_features, x_axis='time', ax=ax)
        #fig.colorbar(img, ax=ax)
        #ax.set(title='MFCC')

        # Einspeichern der aktuellen Auswertung in das Gesamtarray für die spätere Berechnung der globalen Auswertungen
        total_featurearray[:,(i*num_frames):(i+1)*num_frames]=hs_features

        # Berechnen und Einspeichern der Auswertungen in die jeweiligen Featurearrays
        mfcc_mean[:,i]=np.mean(hs_features,axis=1) # Durchschnitt
        mfcc_median[:,i]=np.median(hs_features, axis=1) #Median
        mfcc_stdev[:,i]=np.std(hs_features, axis=1) #Standardabweichung
        mfcc_max[:,i]=np.amax(hs_features, axis=1) #Maximum mfcc
        mfcc_min[:,i]=np.amin(hs_features, axis=1) #Minimum mfcc

    ### Globale Auswertungen

    #ftt

    fft_mean=np.mean(fft_hs,axis=1) #Durchschnitt der FFT Koeffizienten über das Intervall
    #print("fftmean:",fft_mean)

    # Für die mfcc-Features wird das Gesamtfeaturearray verwendet

    mfcc_mean[:,-1]=np.mean(total_featurearray,axis=1) #gloabler Durchschnitt pro mfcc Feature
    mfcc_median[:,-1]=np.median(total_featurearray,axis=1) #globaler Median pro mfcc Feature
    mfcc_stdev[:,-1]=np.std(total_featurearray,axis=1) #globale Standardabweichung pro mfcc Feature
    mfcc_max[:,-1]=np.amax(total_featurearray, axis=1) #globales Maximum pro mfcc Feature
    mfcc_min[:,-1]=np.amin(total_featurearray, axis=1) #globales Minimum pro mfcc Feature
    
    # Für die Auswertungen des Originalsignals reicht das Maximum und Minimum im Array selbst zu finden und ans Ende zu speichern
    # Weitere mögliche Auswertungen: 
    # Mean, median, std der Maximal- und Minimalwerte "Wie gleichförmig waren die Hammerschläge"
    # Länge des Signals, Abstände der Triggersignale "Frequenz der Hammerschläge"
    absmax_mean=np.mean(absmaxarray)
    absmin_mean=np.mean(absminarray)
    absmaxarray[-1]=np.amax(absmaxarray)
    absminarray[-1]=np.amin(absminarray)
    

    ### Aus den Arrays müssen nun eindimensionale Vektoren gemacht werden
    mfcc_mean=mfcc_mean.flatten()
    mfcc_median=mfcc_median.flatten()
    mfcc_stdev=mfcc_stdev.flatten()
    mfcc_max=mfcc_max.flatten()
    mfcc_min=mfcc_min.flatten()
    absmaxarray=absmaxarray.flatten()
    absminarray=absminarray.flatten()
    fft_mean=fft_mean.flatten()

    ### Erstellen des finalen Featurevektors durch aneinanderreihen der eindimensionalen Vektoren in einen sehr langen, eindimensionalen Vektor
    
    selectlist=[]
    for i in range(1,num_mfcc_features+1):
        selectlist.append(i*(num_hs+1)-1)
    #print(selectlist)

    #featurevektor=np.concatenate((mfcc_mean[selectlist],mfcc_median[selectlist],mfcc_stdev[selectlist],mfcc_max[selectlist],mfcc_min[selectlist],np.array(absmaxarray[-1],ndmin=1),np.array(absmax_mean,ndmin=1),np.array(absminarray[-1],ndmin=1),np.array(absmin_mean,ndmin=1),np.array(num_hs,ndmin=1),np.array(num_hs*absmax_mean,ndmin=1)),axis=0)
    featurevektor=np.concatenate((np.array(num_hs,ndmin=1),np.array(sum(absmaxarray)+sum(abs(absminarray)),ndmin=1),np.array(sum(absmaxarray),ndmin=1),np.array(sum(abs(absminarray)),ndmin=1),np.array(absmaxarray[-1],ndmin=1),np.array(abs(absminarray[-1]),ndmin=1),np.array(absmax_mean,ndmin=1),np.array(abs(absmin_mean),ndmin=1),mfcc_mean[selectlist],mfcc_median[selectlist],mfcc_stdev[selectlist],mfcc_max[selectlist],mfcc_min[selectlist]),axis=0)
    ### Aufbau Featurevektor

    # Abfolge im Array: Erst alle means, dann alle medians, usw. jeweils 20*21 Stück: 20 Features*(20 Hammerschläge plus ein Globaler Wert)
    ################### VERALTET  ##################################################
    # Da die Intervalle unterschiedliche Zahl an Hammerschlägen haben, kann keine Einzelschlagauswertung durchgeführt werden
    # 5 Auswertungen für 20 Features für 20 Hammerschläge = 2000 Features
    # 5 globale Werte für 20 Features = 100 Features
    # Abschluss bilden die absoluten Maximalwerte und Minimalwerte der 20 Features, 
    # mit jeweils noch einmal dem höchsten der Maximal/dem kleinsten Minimalwert an 21. Stelle abgespeichert = 42 Features
    # Gesamtlänge featurevektor = 2142 Features
    #Hammerschlag1(mfcc_mean[:,i] concat mfcc_median[:i]), Hammerschlag2(features)
    
    ### Genauer Aufbau:

    #0:mean_feature1_hs1
    #1:mean_feature1_hs2
    #2:...
    #20:mean_feature1_global
    #21:mean_feature2_hs1
    #...
    #420:median_feature1_hs1
    #...
    #2100:abs_max_hs1
    #2101:abs_max_hs2
    #...
    #2120:abs_max
    #2121:abs_min_hs1
    #...
    #2141:abs_min

    ### Umwandeln Vektor zu Liste
    # und an die ID-Liste mit den 6 ID-Werten anhängen



      
    return featurevektor.tolist(),fft_mean.tolist(),signallength, lowcount, signallist,num_frames

##########


#### Aus Feature_list und featurename_list ein Pandas Dataframe erstellen

def signalfeature_dataframe(feature_list,featurename_list):
    signalfeature_df=pd.DataFrame(columns=featurename_list,data=[feature_list])
    print(signalfeature_df)

#sig, trig,signal_ID_list = einlesen_mat_signal(string)
#features, names=generate_signalfeatures(sig, signal_ID_list)
#signalfeature_dataframe(features,names)

#Arbeitspaket 1:
# Erzeugen eines mfcc-Featurevektors mit Featurename und Featurewert
    #features auch über alle Hammerschläge berechnen
    #Matrizen reshapen in (x,1) #5*420 Einträge
    #Erweitern mit Featurename # mean_mfcc_01 #mean_mfcc_total
    #Als Funktion. fct(mat,n_mfcc=10) return (pandas.dataframe)
    #Bitte mehr kommentieren als hier gezeigt

# Diese Funktion durchsucht den Pfad nach Matlab Dateien, um der Funktion "einlesen_mat_signal" dynamisch und automatisch
# die richtige Datei zu übergeben und um alle im Ordner enthaltenen Dateien nacheinander einzulesen
# globalpath ist dabei der übergeordnete Pfad "Datenbasis_clean" 

########## Durchsuchen des angegebenen Ordners nach Matlab-Dateien
# Übergeben wird der Pfad des Ordners
# Ausgegeben wird eine Liste mit Pfaden der Matlab Dateien

def matlab_dateien_finden(globalpath):
    Dateien = []
    for root, dirs, files in os.walk(globalpath):

        for name in files:
            if name.find(".MAT") != -1:
                #print("DATEI: ", name)
                Dateien.append(root+"/"+name)
                
    Anzahl = len(Dateien)
    #print(f"{Anzahl} Dateien gefunden.")
    #print(Dateien)
    return Dateien

#pfaddd = r'G:\06 Studenten\Linke Moritz\Versuchsdaten Entkernen ett\Datenbasis_clean'
#mat = matlab_dateien_finden(pfaddd)
#print(mat)

########## Durchsuchen des angegebenen Ordners nach Entkernfortschritts-Dateien in excel/csv-Format
# Übergeben wird der Pfad des Ordners
# Ausgegeben wird eine Liste mit Pfaden der Entkernfortschritts-Dateien

def entkernfortschrittsdateien_finden(globalpath):
    Dateien = []
        
    for root, dirs, files in os.walk(globalpath):
        for name in sorted(files):
            if name.find(".csv") != -1:
                print("DATEI: ", name)
                Dateien.append(root+"/"+name)
  
    Anzahl = len(Dateien)
    #print(f"{Anzahl} Dateien gefunden.")
    return Dateien

########## Entkernfortschritt auslesen

def Entkernfortschritt_auslesen(filepath): # Argument filepath
    #print(filepath)
    endstring_list=["ent","entkernt","Loch","loch","Ent","Entkernt","entkernt\n"]
    file=open(filepath)
    riegelzahl=file.read().split().count("Riegel")
    #print(riegelzahl)
    file.seek(0)
    data={}
    #f"{x:05d}" #int(1)-->str("01")

    for i in range (1,10):
        newentry={"Riegel 0" + str(i):{"vorne":list(), "hinten":list(),"gewicht":list(),"intervallzahl":int(0)}}
        data.update(newentry)
    for i in range (10,13):
        newentry={"Riegel " + str(i):{"vorne":list(), "hinten":list(),"gewicht":list(),"intervallzahl":int(0)}}
        data.update(newentry)
    gesamtintervallzahl=0
    for key in data:
        file.seek(0)
        for line in file:
                if key in line:
                    line=file.readline()
                    for i in range (1,20):
                        #print(line)
                        if (line.split(sep=";")[i] not in endstring_list):
                            if line.split(sep=";")[i]=="":
                                data[key]["vorne"].append(0)
                            else:
                                data[key]["vorne"].append(int(line.split(sep=";")[i]))
                            data[key]["intervallzahl"]+=1
                            #print(data[key])
                            #print("IZ plus1")
                            gesamtintervallzahl+=1
                        else: 
                            data[key]["vorne"].append(75)
                            break
                    line=file.readline()
                    for i in range (1,len(data[key]["vorne"])):
                        if line.split(sep=";")[i]=="":
                            data[key]["hinten"].append(0)
                        else:
                            data[key]["hinten"].append(int(line.split(sep=";")[i]))
                    data[key]["hinten"].append(75)
                    line=file.readline()
                    for i in range (1,len(data[key]["vorne"])+1):
                        data[key]["gewicht"].append(int(line.split(sep=";")[i]))
    file.close
    return(data)

def ID_Werte_auslesen(path,intervall):
    signal_ID_list=["N/A","N/A","N/A","N/A","N/A","N/A"]
    #print("testa")
    #Werkstoffe gibt es Alu oder Eisen
    if ("Aluminium" in path or "aluminium" in path):
        signal_ID_list[0]="Alumninium"
    elif ("Eisen" in path or "eisen" in path):
        signal_ID_list[0]="Eisen"
    #Riegel-E01-C1_00-20_38400Hz.MAT
    a=path.find("_38400Hz")
    #print(a)
    b=path.rfind("-",0,a)
    #print(b)
    c=path.rfind("_",0,a)
    #print(c)
    #print(path)

    signal_ID_list[1]=path[c-6] #Systembuchstabe A, B, C...
    #print(signal_ID_list[1])
    signal_ID_list[2]=path[c-1] #Chargennummer
    #print(signal_ID_list[2])
    signal_ID_list[3]=path[c-5:c-3] #Riegelnummer
    #print(signal_ID_list[3])
    signal_ID_list[4]=path[c+1:a] #Intervall
    #print("Intervallbezeichnung",signal_ID_list[4])
    #signal_ID_list[5]=int(signal_ID_list[4][0:2])/20+1 #umwandeln des Intervalls in ein Skalar, das die Intervallnummer angibt. Startwert 1
    signal_ID_list[5]=intervall+1
    #print("Intervallzahl:",signal_ID_list[5])
    
    return signal_ID_list


def generate_feature_name_list(num_mfcc_features,systemfeature_list):
    featurename_list=["material", "system", "batch", "bar_number","interval", "interval_number"] #Liste mit den Basis-ID-Namen
    #2 Intervall Startwert Gewicht
    #3 Intervall Startwert Entkerndistanz
    #4 Intervall Fortschritt Gewicht  ###### Hauptzielgröße 1
    #5 Intervall Fortschritt Entkerndistanz ######## Hauptzielgröße 2
    #13 relativer Intervall Fortschritts Gewicht = #4/#7
    #14 relativer Intervall Fortschritts Entkerndistanz =#5/#8
    #6 Riegel Gesamtintervallzahl bis entkernt
    #7 Riegel Gesamtgewichtsverlust(Sandgewicht) bis entkernt
    #8 Riegel Gesamtentkerndistanz bis entkernt
    #9 Riegel Startgewicht
    #10 Riegel Startentkerndistanz
    #11 Riegel durchschnittliche Entkerndistanz pro Intervall absolut
    #12 Riegel durchschnittliches Entkerngewicht pro Intervall absolut
    #11 Riegel durchschnittliche Entkerndistanz pro Intervall relativ
    #12 Riegel durchschnittliches Entkerngewicht pro Intervall relativ

    featurename_list.extend(systemfeature_list)
    '''
    featurename_list.extend(["00_I_start_totalmass_abs","01_I_start_distance_abs", "02_I_start_sandmass",
                                "03_I_start_sandmass_rel_totalsandmass","04_I_start_sandmass_rel_metallmass","05_I_start_sandmass_rel_totalmass",tart
                                "06_I_delta_mass_abs", "07_I_Delta_Distanz_abs", 
                                "08_I_delta_mass_rel_totalmassdelta", "09_I_delta_distance_real_totaldistance",
                                "10_B_n_intervalls_needed_to_decore", 
                                "11_B_start_mass_abs", "12_B_start_distance_abs","13_B_sandmass",
                                "14_B_sandmass_rel_totalmass","15_B_sandmass_rel_metalmass",
                                "16_B_average_mass_per_I_abs","17_B_average_distance_per_I_abs",
                                "18_B_average_mass_per_I_rel_totalmassdelta","19_B_average_distance_per_I_rel_totaldistance"
                            ])
    
    #allgemeine Infos
    featurename_list.append("20_I_n_hammerblows")
    featurename_list.append("21_I_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow")
    

    #Dann kommen ein globaler Maximalwert und der Durchschnittswert der Hammerschlag-Maxima für das Originalsignal
    featurename_list.append("22_I_sum_of_max_positive_acceleration_of_each_hammerblow")
    featurename_list.append("23_I_sum_of_max_negative_acceleration_of_each_hammerblow")
    featurename_list.append("24_I_max_positive_acceleration_of_all_hammerblows")
    featurename_list.append("25_I_max_negative_acceleration_of_all_hammerblows")
    featurename_list.append("26_I_mean_of_max_positive_acceleration_of_each_hammerblow")
    featurename_list.append("27_I_mean_of_max_negative_acceleration_of_each_hammerblow")    
    #Dann kommen ein globaler Minimalwert und der Durchschnittswert der Hammerschlag-Minima für das Originalsignal

    #allgemeine Infos
    featurename_list.append("20_i_n_hammerblows")
    featurename_list.append("21_i_sum_of_abs_max_negative_and_positive_acceleration_of_each_hammerblow")
    
    #Dann kommen ein globaler Maximalwert und der Durchschnittswert der Hammerschlag-Maxima für das Originalsignal
    featurename_list.append("22_i_sum_of_max_positive_acceleration_of_each_hammerblow")
    featurename_list.append("23_i_sum_of_max_negative_acceleration_of_each_hammerblow")
    featurename_list.append("24_i_max_positive_acceleration_of_all_hammerblows")
    featurename_list.append("25_i_max_negative_acceleration_of_all_hammerblows")
    featurename_list.append("26_i_mean_of_max_positive_acceleration_of_each_hammerblow")
    featurename_list.append("27_i_mean_of_max_negative_acceleration_of_each_hammerblow")    
    #Dann kommen ein globaler Minimalwert und der Durchschnittswert der Hammerschlag-Minima für das Originalsignal

    '''
    featurename_list.extend(["00_i_s_m_total","01_i_s_d_progress", "02_i_s_m_sand",
                                "03_i_s_m_progress","04_i_s_m_sand_r_metall","05_i_s_m_sand_r_total",
                                "06_i_m_progress", "07_i_d_progress", 
                                "08_i_m_progress_r", "09_i_d_progress_r",
                                "10_b_n_i_needed", 
                                "11_b_s_m", "12_b_s_d","13_b_m_sand",
                                "14_b_m_sand_r_total","15_b_mass_r_metal",
                                "16_b_m_average_per_i","17_b_d_average_per_i",
                                "18_b_m_average_progress","19_b_d_average_progress"
                            ])
    
    #allgemeine Infos
    featurename_list.append("20_i_n_hammerblows")
    featurename_list.append("21_i_a_sum_abs_max_posneg")
    
    #Dann kommen ein globaler Maximalwert und der Durchschnittswert der Hammerschlag-Maxima für das Originalsignal
    featurename_list.append("22_i_a_sum_max_pos")
    featurename_list.append("23_i_a_sum_max_neg")
    featurename_list.append("24_i_a_max_pos")
    featurename_list.append("25_i_a_max_neg")
    featurename_list.append("26_i_a_mean_max_pos")
    featurename_list.append("27_i_a_mean_max_neg")    
    #Dann kommen ein globaler Minimalwert und der Durchschnittswert der Hammerschlag-Minima für das Originalsignal

    


    #Als erstes kommen die 5 Auswertungen der Features
    #die Schleife zählt die Features durch
    for i in range(0,num_mfcc_features):
        featurename_list.append("mean_mfcc"f"{i+1:02d}")

    #Das gleiche für den Median...
    for i in range(0,num_mfcc_features):
        featurename_list.append("median_mfcc"f"{i+1:02d}")
    
    # und die Standardabweichung...
    for i in range(0,num_mfcc_features):
        featurename_list.append("stdev_mfcc"f"{i+1:02d}")
    
    #und für das Maximum...
    for i in range(0,num_mfcc_features):
        featurename_list.append("max_mfcc"f"{i+1:02d}")
    
    #und für das Minimum...
    for i in range(0,num_mfcc_features):
        featurename_list.append("min_mfcc"f"{i+1:02d}")

    for i in range(4,19201,4):
        featurename_list.append("fft_"f"{i:05d}_Hz")

    #print(featurename_list)
    return featurename_list

def intervallanzahl_Riegel_auslesen(data):
    keys = list(data.keys())
    Anzahl_Riegel = len(keys)
    Riegel_Intervallanzahl = []
    
    for key in data:
        Riegel_Intervallanzahl.append(int(data[key]['intervallzahl']))

    return Riegel_Intervallanzahl

def gesamttabelle_auslesen(filepath):
    path = os.path.join(filepath,'gesamttabelle_NaN_newnames.csv')
    gesamttabelle = pd.read_csv(path, sep=';',decimal=".",header=0,index_col=0)
    #gesamttabelle.set_index('System')
    #print(gesamttabelle)
    return gesamttabelle


def Riegelintervallauslesen(data,Riegelnummer,Intervallnummer):
    keys = list(data.keys())
    Intervallnummer=Intervallnummer+1
    riegeldata=data[keys[Riegelnummer]]
    intervall_featurelist=[]

    #0 Intervall Startwert Gewicht
    intervall_featurelist.append(riegeldata["gewicht"][Intervallnummer-1])
    #1 Intervall Startwert Distanz Entkernfortschritt
    intervall_featurelist.append((riegeldata["vorne"][Intervallnummer-1]+riegeldata["hinten"][Intervallnummer-1])/150)
    #2 Intervall Startwert Sandgewicht
    intervall_featurelist.append(riegeldata["gewicht"][Intervallnummer-1]-riegeldata["gewicht"][-1])
    #3 Intervall Startwert Sandgewicht relativ zum Gesamtsandgewicht
    intervall_featurelist.append((riegeldata["gewicht"][Intervallnummer-1]-riegeldata["gewicht"][-1])/(riegeldata["gewicht"][0]-riegeldata["gewicht"][-1]))
    #4 Intervall Startwert Sandgewicht zu Metallgewicht
    intervall_featurelist.append((riegeldata["gewicht"][Intervallnummer-1]-riegeldata["gewicht"][-1])/riegeldata["gewicht"][-1])
    #5 Intervall Startwert Sandgewicht zu Startwert Gesamtgewicht
    intervall_featurelist.append((riegeldata["gewicht"][Intervallnummer-1]-riegeldata["gewicht"][-1])/riegeldata["gewicht"][Intervallnummer-1])

    #6 Intervall Fortschritt Gewicht absolut ###### Hauptzielgröße 1
    intervall_featurelist.append(riegeldata["gewicht"][Intervallnummer-1]-riegeldata["gewicht"][Intervallnummer])
    #7 Intervall Fortschritt Distanz absolut ######## Hauptzielgröße 2
    intervall_featurelist.append(riegeldata["vorne"][Intervallnummer]+riegeldata["hinten"][Intervallnummer]-intervall_featurelist[1])
    #8 Intervall Fortschritt Gewicht relativ ######## Hauptzielgröße 3
    intervall_featurelist.append(intervall_featurelist[6]/(riegeldata["gewicht"][0]-riegeldata["gewicht"][-1]))
    #9 Intervall Fortschritt Distanz relativ ######## Hauptzielgröße 4
    intervall_featurelist.append(intervall_featurelist[7]/(riegeldata["vorne"][-1]-riegeldata["vorne"][0]+riegeldata["vorne"][-1]-riegeldata["hinten"][0]))

    #10 Riegel Gesamtintervallzahl bis entkernt
    intervall_featurelist.append(riegeldata["intervallzahl"])

    #11 Riegel Startgewicht
    intervall_featurelist.append(riegeldata["gewicht"][0])
    #12 Riegel Startentkerndistanz
    intervall_featurelist.append(riegeldata["vorne"][0]+riegeldata["hinten"][0])

    #13 Riegel Sandgewicht
    intervall_featurelist.append(riegeldata["gewicht"][0]-riegeldata["gewicht"][-1])
    #14 Riegel Sandgewicht zu Gesamtgewicht
    intervall_featurelist.append((riegeldata["gewicht"][0]-riegeldata["gewicht"][-1])/(riegeldata["gewicht"][0]))
    #15 Riegel Sandgewicht zu Metallgewicht
    intervall_featurelist.append((riegeldata["gewicht"][0]-riegeldata["gewicht"][-1])/(riegeldata["gewicht"][-1]))

    #16 Riegel durchschnittliches Entkerngewicht pro Intervall absolut
    intervall_featurelist.append(riegeldata["gewicht"][0]-riegeldata["gewicht"][-1]/riegeldata["intervallzahl"])
    #17 Riegel durchschnittliche Entkerndistanz pro Intervall absolut
    intervall_featurelist.append(150/riegeldata["intervallzahl"])
    #18 Riegel durchschnittliches Entkerngewicht pro Intervall relativ
    intervall_featurelist.append(1/riegeldata["intervallzahl"])
    #19 Riegel durchschnittliche Entkerndistanz pro Intervall relativ
    intervall_featurelist.append(1/riegeldata["intervallzahl"])

    return intervall_featurelist 
########## 


path = r'D:/EntkernKI'
path=path+"/Datenbasis_clean"
num_mfcc_features=5
pd_systemfeatures=gesamttabelle_auslesen(path)
column_name_list=generate_feature_name_list(num_mfcc_features, pd_systemfeatures.columns)
fortschrittcsv_list = entkernfortschrittsdateien_finden(path)

data_list=[]
signallength_list=[]
sum_lowcount5000=0
sum_lowcount4000=0
sum_lowcount3000=0
sum_lowcount2000=0
sum_lowcount1000=0
sum_lowcount750=0
sum_lowcount500=0
sum_lowcount250=0
sum_lowcount100=0
sum_lowcountbigger5000=0
sum_count=0
global_signallist=[]

print("CSV-Anzahl: ", len(fortschrittcsv_list))
print("CSV-Anzahl: ", len(fortschrittcsv_list))
for i in fortschrittcsv_list:
    print(i)
    mat_signalfile_list=matlab_dateien_finden(os.path.dirname(i))
    print("ZAHL AN GEFUNDENEN MATLABDATEIEN IN CSV:", len(mat_signalfile_list))
    fortschritts_data=Entkernfortschritt_auslesen(i)
    keys = list(fortschritts_data.keys())
    intervallgesamt=0
    for key in keys:
        intervallgesamt=intervallgesamt+fortschritts_data[key]["intervallzahl"]
    print("ZAHL AN INTERVALLEN IN CSV:", intervallgesamt)
    Anzahl_Riegel = len(keys)
    intervallanzahl_list=intervallanzahl_Riegel_auslesen(fortschritts_data)
    l=0
    print("Riegelzahl: ", Anzahl_Riegel)
    for j in range(0,Anzahl_Riegel):
        print("Intervallanzahl: ", intervallanzahl_list[j])
        for k in range(0,intervallanzahl_list[j]):
            ID_list=[]
            systemfeatures=[]
            print("DURCHLAUF", i, j, k)
            #print(mat_signalfile_list[l])
            ID_list.extend(ID_Werte_auslesen(mat_signalfile_list[l],k))
            #print("test1")
            signalfeatures, fft_mean, signallength, lowcount, signallist,num_mfcc_frames=generate_signalfeatures(mat_signalfile_list[l])
            sum_lowcount5000=sum_lowcount5000+lowcount[0]
            sum_lowcount4000=sum_lowcount4000+lowcount[1]
            sum_lowcount3000=sum_lowcount3000+lowcount[2]
            sum_lowcount2000=sum_lowcount2000+lowcount[3]
            sum_lowcount1000=sum_lowcount1000+lowcount[4]
            sum_lowcount750=sum_lowcount750+lowcount[5]
            sum_lowcount500=sum_lowcount500+lowcount[6]
            sum_lowcount250=sum_lowcount250+lowcount[7]
            sum_lowcount100=sum_lowcount100+lowcount[8]
            sum_lowcountbigger5000=sum_lowcountbigger5000+lowcount[9]
            sum_count=sum_count+lowcount[10]
            signallength_list.append(signallength)
            global_signallist=global_signallist + signallist
            l=l+1
            #print("test2")
            systemfeatures.extend(pd_systemfeatures.loc[ID_list[1]].tolist())
            #print("test3")
            fortschrittfeatures=Riegelintervallauslesen(fortschritts_data, j, k)
            #print(len(ID_list))
            #print(len(signalfeatures))
            #print(len(systemfeatures))
            #print(len(fortschrittfeatures))
            #print("test4")
            data_list.append(ID_list+systemfeatures+fortschrittfeatures+signalfeatures+fft_mean)
            

print("Signale mit Amplitude unter 5000:", sum_lowcount5000)
print("Signale mit Amplitude unter 4000:", sum_lowcount4000)
print("Signale mit Amplitude unter 3000:", sum_lowcount3000)
print("Signale mit Amplitude unter 2000:", sum_lowcount2000)
print("Signale mit Amplitude unter 1000:", sum_lowcount1000)
print("Signale mit Amplitude unter 750", sum_lowcount750)
print("Signale mit Amplitude unter 500:", sum_lowcount500)
print("Signale mit Amplitude unter 250:", sum_lowcount250)
print("Signale mit Amplitude unter 100:", sum_lowcount100)
print("Signale mit Amplitude über 5000:", sum_lowcountbigger5000)
print("Gezählte Signale insgesamt:", sum_count)

print("Maximale Signallänge: ", max(signallength_list))
print("Minimale Signallänge: ", min(signallength_list))
print("Mean Signallänge ", mean(signallength_list))
print("Median Signallänge ", np.median(signallength_list))
#print(data_list)
#print(column_name_list)
pd_dataset=pd.DataFrame(columns=column_name_list,data=data_list)



pd_dataset.to_pickle(path+r"\DATASET_complete_v12.pickle")
pd_dataset.to_csv(path+r"\DATASET_complete_v12.csv")

#hotencoding
#print(pd_dataset)
y = pd.get_dummies(pd_dataset.system, prefix='system')
#print(y)
pd_dataset=pd.concat([y,pd_dataset],axis=1)
#print(pd_dataset)
y = pd.get_dummies(pd_dataset.sand, prefix='sand')
#print(y)
pd_dataset=pd.concat([y,pd_dataset],axis=1)
#data.insert(2,y.columns(name))
#print(pd_dataset)
y = pd.get_dummies(pd_dataset.binder, prefix='binder')
#print(y)
pd_dataset=pd.concat([y,pd_dataset],axis=1)
#data.insert("Binder",y.columns,y.values)
#print(pd_dataset)


pd_dataset.drop('interval',axis=1,inplace=True)
pd_dataset.drop('binder',axis=1,inplace=True)
pd_dataset.drop('material',axis=1,inplace=True)
pd_dataset.drop('sand',axis=1,inplace=True)

pd_dataset.to_pickle(path+r"\DATASET_INTERVALL_he_v12.pickle")
pd_dataset.to_csv(path+r"\DATASET_INTERVALL_he_v12.csv")

print(pd_dataset)
for i in range(0,len(signallist)):
    plt.plot(range(0,len(signallist[i])),signallist[i])
#plt.show()


pd_dataset.drop('interval_number',axis=1,inplace=True)
pd_dataset.drop(["00_i_s_m_total","01_i_s_d_progress", "02_i_s_m_sand",
                 "03_i_s_m_progress","04_i_s_m_sand_r_metall","05_i_s_m_sand_r_total",
                 "06_i_m_progress", "07_i_d_progress",
                 "08_i_m_progress_r", "09_i_d_progress_r"],
                axis=1,inplace=True)


####Folgendes muss noch angepasst werden

newdata=pd.DataFrame()
for i in pd_dataset["system"].unique():
    print("system",i)
    for j in pd_dataset.loc[(pd_dataset['system'] == i)]['batch'].unique():
        print('batch', j)
        for k in pd_dataset.loc[(pd_dataset['system'] == i)&(pd_dataset['batch'] == j)]['bar_number'].unique():
            print("Riegel",k)

            newdata1=pd.DataFrame([pd_dataset.loc[(pd_dataset['system'] == i)&(pd_dataset['batch'] == j)&(pd_dataset['bar_number'] == k),pd_dataset.columns[0:101]].iloc[0]]).reset_index(drop=True) #bleibt gleich
            newdata2=pd.DataFrame([pd_dataset.loc[(pd_dataset['system'] == i)&(pd_dataset['batch'] == j)&(pd_dataset['bar_number'] == k),pd_dataset.columns[101:105]].iloc[0]]).reset_index(drop=True) #bleibt gleich
            newdata1=pd.concat([newdata1,newdata2],axis=1)
            
            newdata2=pd_dataset.loc[(pd_dataset['system'] == i)&(pd_dataset['batch'] == j)&(pd_dataset['bar_number'] == k),pd_dataset.columns[105:111]].sum(axis=0).to_frame().T.reset_index(drop=True) #Summenbildung Hammerschläge und Beschleunigungssumme positive werte
            newdata1=pd.concat([newdata1,newdata2],axis=1)

            newdata2=pd_dataset.loc[(pd_dataset['system'] == i)&(pd_dataset['batch'] == j)&(pd_dataset['bar_number'] == k),pd_dataset.columns[111:113]].mean(axis=0).to_frame().T.reset_index(drop=True) #Durchschnittsbildung Beschleunigungsdurchschnitt maxwerte positiv
            newdata1=pd.concat([newdata1,newdata2],axis=1)

            newdata2=pd_dataset.loc[(pd_dataset['system'] == i)&(pd_dataset['batch'] == j)&(pd_dataset['bar_number'] == k),pd_dataset.columns[113:4938]].mean(axis=0).to_frame().T.reset_index(drop=True) #Durchschnittsbildung mfcc und fft Werte
            newdata1=pd.concat([newdata1,newdata2],axis=1)

            newdata1.rename(columns={"10_b_n_i_needed":"00_b_n_i_needed",
                                    "11_b_s_m" : "01_b_s_m",
                                    "12_b_s_d" : "02_b_s_d",
                                    "13_b_m_sand" : "03_b_m_sand",
                                    "14_b_m_sand_r_total" : "04_b_m_sand_r_total",
                                    "15_b_mass_r_metal" : "05_b_mass_r_metal",
                                    "16_b_m_average_per_i" : "06_b_m_average_per_i",
                                    "17_b_d_average_per_i" : "07_b_d_average_per_i",
                                    "18_b_m_average_progress" : "08_b_m_average_progress",
                                    "19_b_d_average_progress" : "09_b_d_average_progress",
                                    "20_i_n_hammerblows" : "10_b_n_hammerblows",
                                    "21_i_a_sum_abs_max_posneg" : "11_b_a_sum_abs_max_posneg",  #sum(sum(max(hammerblow)+abs(min(hammerblow)) all hammerblows intervall)) of all intervalls)
                                    "22_i_a_sum_max_pos" : "12_b_a_sum_max_pos",  #sum(sum(max(hammerblow) all hammerblows intervall)) of all intervalls)
                                    "23_i_a_sum_max_neg" : "13_b_a_sum_max_neg", #sum(sum(minhammerblow) all hammerblows intervall)) of all intervalls)
                                    "24_i_a_max_pos" : "14_b_a_max_pos", #max(max(hammerblows in intervall) of all intervalls)
                                    "25_i_a_max_neg" : "15_b_a_max_neg", #min(min(hammerblows in intervall) of all intervalls)
                                    "26_i_a_mean_max_pos" : "16_b_a_mean_max_pos", #mean(mean(hammerblows in intervall) of all intervalls)
                                    "27_i_a_mean_max_neg" : "17_b_a_mean_max_neg" #mean(mean(hammerblows in intervall) of all intervalls)
                                    },inplace=True)
            newdata=pd.concat([newdata, newdata1],ignore_index=True)

                            
print(newdata)
newdata.to_pickle(path+r"\DATASET_RIEGEL_he_v12.pickle")
newdata.to_csv(path+r"\DATASET_RIEGEL_he_v12.csv")




