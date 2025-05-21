default:
    @just --list

demo-tape:
    vhs demo/demo.tape

demo-docker-build:
    docker build -t demo-langchain -f demo/Dockerfile .

demo-docker-run:
    docker run -ti --rm --name demo-langchain demo-langchain
