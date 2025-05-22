mod examples-basic './examples/basic'

default:
    @just --list

format:
    #poetry run ruff format
    make format

lint:
    -time poetry run ty check
    make lint
