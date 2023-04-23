# text-mining-search-engine
Search engine application to index and retrieve documents

# Prerequisites
1. Clone this repository using the command `clone https://github.com/pramodatre/text-mining-search-engine.git`
2. Create Python environment on conda for this project (e.g., `conda crate --name udemytextmining python=3.9`) and install the the dependencies by running the following command `pip install -r requirements.txt`
3. Install Docker and Docker Compose on your machine by following the instructions here:
* [MacOS](https://docs.docker.com/desktop/install/mac-install/)
* [Windows](https://docs.docker.com/desktop/install/windows-install/)
* [Linux](https://docs.docker.com/desktop/install/linux-install/)

# Search Engine
## Flask application
After you clone the repository, from project home directory (`text-mining-search-engine`), launch the flask application using the following command.
```
flask --app app run
```
Once flask brings up the service, you will be able access the search engine at https://localhost:5000

## Containerized application
For your convenience and for deploying the search engine as a web application, the entire search engine is containerized using Docker. You need to have Docker and Docker Compose installed on your machine before you can run the following commands. 
```
docker compose -f docker-compose.yml build
```
Once you have build the container file, you can bring up the service using the following command.
```
docker compose -f docker-compose.yml up
```
Once docker compose brings up the service, you will be able access the search engine at https://localhost:8000