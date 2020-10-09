import pandas as pd
import matplotlib.pyplot as plt

# Other  
import librosa
import librosa.display
import numpy as np
import os
import sys
import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

nameFile = 'data.csv'

data = 'Audio_Speech_Actors_01-24/'
dir_list = os.listdir(data)
dir_list.sort()

data_tess = 'TESS_Toronto_emotional_speech_set_data/'
dir_list_tess = os.listdir(data_tess)

print('Prepara la data RADVESS')
emotion = []
path    = []

for i in dir_list:
    fname = os.listdir(data + i) # lista los archivos del directorio
    for f in fname: #recorre cada archivo
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2])) #emocion 
        path.append(data + i + '/' + f) # actor__/nombre del archivo

# dataframe of files
emotion_df  = pd.DataFrame(emotion, columns=['Emocion'])
path_df     = pd.DataFrame(path, columns=['Path'])
clase_df    = pd.DataFrame(emotion, columns=['Clase'])

RAV_df      = pd.concat([emotion_df, path_df, clase_df], axis=1)

# changing integers to actual emotions.
RAV_df.Emocion.replace({1:'neutral', 2:'calma', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, 
                        inplace=True)

# identificar la clase 
RAV_df.Clase.replace({1:'negativo', 2:'positivo', 3:'positivo', 4:'negativo', 5:'negativo', 6:'negativo', 7:'negativo', 8:'positivo'}, inplace=True)


print(RAV_df.head())
print(RAV_df.shape)
print(RAV_df.groupby('Clase').size())

print('Prepara la data TESS')

emotion = []
path    = []
for dir in dir_list_tess:
    directories = os.listdir(data_tess + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        
        if part=='ps':
            emotion.append('surprise')
        else:
            emotion.append(part)
        path.append(data_tess + dir + '/' + file)

# dataframe of files
emotion_df  = pd.DataFrame(emotion, columns=['Emocion'])
path_df     = pd.DataFrame(path, columns=['Path'])
clase_df    = pd.DataFrame(emotion, columns=['Clase'])

Tess_df = pd.concat([emotion_df, path_df, clase_df], axis=1)

Tess_df.Clase.replace({'angry':'negativo', 'disgust':'negativo', 'fear':'negativo', 'happy':'positivo', 'neutral':'negativo', 'surprise':'positivo', 'sad':'negativo'}, inplace=True)

print('TESS')
print(Tess_df.head())
print(Tess_df.shape)
print(Tess_df.groupby('Clase').size())

df = pd.concat([RAV_df,Tess_df], axis = 0)
print('df')
print(df.head())
print(df.shape)

print('group')
print(df.groupby('Clase').size())

#Guardar archivo
df.to_csv(nameFile,index=False)

######################################################################
#                 Data Augmentation                                  #
######################################################################

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

######################################################################
#                 Extraccion de caracteristicas                      #
######################################################################

def extract_features(data):
    print('extract_features')
    print('sample')
    print(sample_rate)
    
    result = np.array([])
    
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result

print('Apertura del archivo')
ref = pd.read_csv(nameFile)
print(ref.head())

# Note this takes a couple of minutes (~10 mins) as we're iterating over 4 datasets  
dfn = pd.DataFrame(columns=['feature'])
df_noise = pd.DataFrame(columns=['feature'])
df_speedpitch = pd.DataFrame(columns=['feature'])

# loop feature extraction over the entire dataset
counter=0
for index, path in enumerate(ref.Path):
    print('######### audio ######### ', counter)
    print(path)
    # exit()
    X, sample_rate = librosa.load(path
                                  , res_type='kaiser_fast'
                                  ,duration=2.5
                                  ,sr=44100
                                  ,offset=0.5
                                 )
    
    sample_rate = np.array(sample_rate)

    # without augmentation
    resul = extract_features(X)
    result = np.array(resul)
    dfn.loc[counter] = [result]
    
    # data with noise
    noise_data = noise(X)
    res2 = extract_features(noise_data)
    df_noise.loc[counter] = [res2]
    # result = np.vstack((result, res2)) # stacking vertically

    # # data with stretching and pitching
    stretch_data = stretch(X)
    data_stretch_pitch = pitch(stretch_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    df_speedpitch.loc[counter] = [res3]
    # result = np.vstack((result, res3)) # stacking vertically
    
    counter=counter+1
    
# Check a few records to make sure its processed successfully
print('Check a few records to make sure its processed successfully')
print('dfn: ', dfn.shape)
print('df_noise: ', df_noise.shape)
print('df_speedpitch: ', df_speedpitch.shape)

# Now extract the mean bands to its own feature columns
print('Now extract the mean bands to its own feature columns')
df_noise = pd.concat([ref,pd.DataFrame(df_noise['feature'].values.tolist())],axis=1)
df_speedpitch = pd.concat([ref,pd.DataFrame(df_speedpitch['feature'].values.tolist())],axis=1)

df = pd.concat([ref,pd.DataFrame(dfn['feature'].values.tolist())],axis=1)
print(df)

print('mostrar las caracteristicas')
df = pd.concat([df,df_noise,df_speedpitch],axis=0)
print(df.head())

# replace NA with 0
df=df.fillna(0)
print('replace NA with 0')
print(df.shape)

df.to_csv(nameFile,index=False)
print('save: ', nameFile)
