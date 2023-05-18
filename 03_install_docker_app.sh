# file: install_docker_app.sh
# functie: maak een omgeving in docker
#          Na de installatie wordt er
#          url gegenereerd waareem de omgeving opgestart kan worden
#          Deze url is nu http://127.0.0.1:8888/lab?token=616ae4cc2592d5aa9113d310ca079678e4cff1ed2aed24b4
#
#          Alternatief
#          Kijk in visual code mbv de docker plugin naar de aangemaakt container

# sudo docker-compose -f "docker-compose.yml" -d --build up 
sudo docker-compose --env-file ./sample.env -f "docker-compose.yml" up 
