# alusive_fastapi_server


to build the project using docker, 
use

1. `docker build -t fastapi-app .`
2. `docker run -p 80:80 fastapi-app` 


### to install packages in the docker image
1. Start a Container from the Image 
    This starts a new container and opens a shell inside it.
    `docker run -it --name fastapi-container fastapi-app bash`

2. Install Packages Inside the Container
    Once inside the container, you can install packages. Example for apt-based systems:
    `apt update && apt install -y package1 package2 `

    Eg: install python-poppler
    `apt update && apt install poppler-utils` 


3. Commit the Changes (If Needed)
    After installing psackages, you might want to commit the changes to the image.
    `docker commit fastapi-container fastapi-app-updated`

    Now, you have a new image fastapi-app-updated with the installed packages.

4.  Run a Shell in an Existing Container
    If you have an existing container, you can run a shell inside it.
    ```
        docker start fastapi-container
        docker exec -it fastapi-container bash
    ```





However,  you neeed to install poppler on your mac using this
`brew install poppler`