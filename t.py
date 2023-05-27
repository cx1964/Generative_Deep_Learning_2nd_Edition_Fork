# file: t.py 
# functie: script tbv uitzoeken van fout in notebook transformer.aangepast.ipynb 

# cel1
import os
import glob
import numpy as np
import time
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks

import music21

from transformer_utils import (
    parse_midi_files,
    load_parsed_files,
    get_midi_note,
    SinePositionEncoding,
)

# cel2
PARSE_MIDI_FILES = True
#PARSED_DATA_PATH = "/app/notebooks/11_music/01_transformer/parsed_data/"
PARSED_DATA_PATH = "/home/claude/Documents/sources/python/python3/AI/Generative_Deep_Learning_2nd_Edition_Fork/notebooks/11_music/01_transformer/parsed_data/"
DATASET_REPETITIONS = 1

SEQ_LEN = 50
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 5
DROPOUT_RATE = 0.3
FEED_FORWARD_DIM = 256
LOAD_MODEL = False

# optimization
EPOCHS = 5000
BATCH_SIZE = 256

GENERATE_LEN = 50

# toegevoegd
# Instellen van de Environment
# Zie: https://web.mit.edu/music21/doc/usersGuide/usersGuide_24_environment.html#usersguide-24-environment
env = music21.environment.UserSettings()
env.delete()
env.create()
# set environmment
env['autoDownload'] = 'allow'
env['lilypondPath'] = '/usr/bin/lilypond'
#env['musescoreDirectPNGPath'] = 'musescore.mscore'  # Deze variant gebruiken indien musescore als package geinstalleerd is
# tbv musescore4
env['musescoreDirectPNGPath'] = '/home/claude/Applications/MuseScore-4.0.2.230651545-x86_64_fad3de9b129dfdf8ddca9d6b70d0e1a3.appimage' # Deze variant gebruiken indien musescore als appimage is geinstalleerd
# tbv musescore3
#env['musescoreDirectPNGPath'] = '/home/claude/Applications/MuseScore-3.6.2.548021370-x86_64_461d9f78f967c0640433c95ccb200785.AppImage' # Deze variant gebruiken indien musescore als appimage is geinstalleerd
# env['musicxmlPath'] = '/usr/bin/musescore3'  # Deze variant gebruiken indien musescore als package geinstalleerd is
print('Environment settings:')
print('lilypond: ', env['lilypondPath'])
print('musicXML: ', env['musicxmlPath'])
print('musescore: ', env['musescoreDirectPNGPath'])
print("Env ok")
print('Music21 version', music21.VERSION_STR) 

#cel3
# Load the data
file_list = glob.glob("/home/claude/Documents/Data/Generative_Deep_Learning_2nd_Edition/bach-cello/*.mid")
print(f"Found {len(file_list)} midi files")

#cel4
parser = music21.converter

#cel5
print("file_list:", file_list)
example_score = (
    music21.converter.parse(file_list[1]).splitAtQuarterLength(12)[0].chordify()
    # music21.converter.parse("/home/claude/Documents/Data/Generative_Deep_Learning_2nd_Edition/bach-cello/cs1-1pre.mid", format='midi').splitAtQuarterLength(12)[0].chordify()
    # music21.converter.parse("/home/claude/Documents/Data/Generative_Deep_Learning_2nd_Edition/bach-cello/Prelude_No20_C_Minor_Frederic_Chopin_v000c_zonder_bas.musicxml", format='musicxml').splitAtQuarterLength(12)[0].chordify()
)

#cel6
# Dit geeft een fout als men notebook draait vanuit visual code
# Als men de code van cel1 t/m cel6 draait als python script buiten visual code
# dan werkt het gewoon
example_score.show()