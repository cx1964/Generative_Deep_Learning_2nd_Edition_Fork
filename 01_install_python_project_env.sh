#!/bin/bash
# file:  install_python_project_env.sh
# functie: install python project environment obv pyenv
# opmerking: Voor dat men dit script draait, check:
#            - alle benodigde modules worden genoemd om te installeren in ### Begin aanpassen ###
#              Gebruik eventule de specificatie tav welke packages te installeren op de aanwezigheid van de file requirements.txt
#            - Of de laatste versie van python is ingesteld.
#              zo nee pas die eventueel aan  
#            - Of dat men in de project directory staat als men dit script draait
#
#            Dit script is om een python environment te maken
#            zodat geen gebruik van docker container hoeft gemaakt te worden
#            om de benodigde (python) omgeving in te maken.
#
# documentatie: https://www.activestate.com/resources/quick-reads/how-to-manage-python-dependencies-with-virtual-environments/
#               mbt bash https://linuxconfig.org/bash-scripting-tutorial-for-beginners


### Begin aanpassen ###
# kies een python versie mbv
# pyenv install --list
# Nog onbekend hoe een versie te installeren hoger dan de system python versie op het OS
export PYTHON_VERSION="3.10.6" # 20230514 hoogste versie aanwezig op bijgswerkte Ubuntu 22.04 versie
#export PYTHON_VERSION="3.10-buster" 
### Einde aanpassen ###


# Dit script gaat ervan uit dat pyenv al is geinstalleerd
# als pyenv is geinstalleerd moet de directory ~/.pyenv bestaan
# nog een conditie maken dat dit script alleen iets doet als de directory ~/.pyenv bestaat.
# anders een melding geven dat men pyenv moet installeren
# Zie dan https://www.activestate.com/resources/quick-reads/how-to-manage-python-dependencies-with-virtual-environments/
# en gebruik paragraaf How to Install Pyenv


# Waar sta ik?
pwd

read -p 'Sta je in de project Directory (j/n)?:' ;
echo "";

if [ ${REPLY} != "j" ]; then
  # Breek installatie af omdat men NIET in de project directory staat
  echo "Abort installatie"
  return 1
else
  # Do de installatie omdat men de project directory staat

  # update pipenv 
  pip install --upgrade pipenv

  # install python version
  pipenv --python $PYTHON_VERSION

  pipenv install jupyter
  
  ### Begin aanpassen ###
  # welke pipenv install packages geinstalleerd moet worden obvb requirements.txt
  pipenv install matplotlib==3.6.3
  pipenv install jupyterlab==3.5.3
  pipenv install scipy==1.10.0
  pipenv install scikit-learn==1.2.1
  pipenv install scikit-image==0.19.3
  pipenv install pandas==1.5.3
  pipenv install music21==8.1.0
  pipenv install black[jupyter]==23.1.0
  pipenv install click==8.0.4
  pipenv install flake8==5.0.4
  pipenv install kaggle==1.5.12
  pipenv install pydot==1.4.2
  pipenv install ipywidgets==8.0.4
  pipenv install tensorflow==2.10.1
  pipenv install tensorboard==2.10.1
  pipenv install tensorflow_probability==0.18.0
  pipenv install flake8-nb==0.5.2
    # voorbeeld specifieke versie: pipenv install 'setuptools==62.1.0'
  ## Einde aanpassen ###

  # Toon geinstalleerde modules 
  ###pipenv run pip list
  return 0
fi

echo "start nu pipenv shell"
echo "voor het activeren van de virtual env van het project"