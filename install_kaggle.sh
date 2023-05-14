# file: install_kaggle.sh
# functie: install kaggle.json file in ~/.kaggle/
# Opmerking: Als men vanuit je kaggle account
#            mbv generarte api een api key genereerd
#            wordt er een kaggel.json file aangemaakt in ~/Downloads.
#            Deze file moet geplaats worden in ~/.kaggle/ subdirectory.
#            Dit script zorgt daarvoor
# Doc: zie https://github.com/Kaggle/kaggle-api

# create directory
cd ~
mkdir .kaggle
# copy the generated json file
cp ~/Downloads/kaggle.json ~/.kaggle/ 