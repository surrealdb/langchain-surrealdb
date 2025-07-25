mod examples-basic './examples/basic'
mod examples-graph './examples/graph'
mod examples-graphrag './examples/graphrag-travel-group-chat'

default:
    @just --list

format:
    #poetry run ruff format
    make format

lint:
    -time poetry run ty check
    make lint

# Install dependencies
install:
    poetry update --with lint,typing,test
    cd examples/basic && poetry update
    cd examples/graph && poetry update && poetry install
