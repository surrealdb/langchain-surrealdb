mod examples-basic './examples/basic'
mod examples-graph './examples/graph'
mod examples-graphrag './examples/graphrag-travel-group-chat'

default:
    @just --list

format:
    #uv run ruff format
    make format

lint:
    -time uv run ty check
    make lint

# Install dependencies
install:
    uv sync --all-groups --all-extras
    cd examples/basic && uv sync
    cd examples/graph && uv sync

example-graphrag *ARGS:
    uv run --package graphrag-travel-group-chat cli {{ARGS}}
