#!/bin/bash

# docker run -d --name postgres_bbot \
#     --rm \
#     -p 5432:5432 \
#     -v $(pwd)/data:/var/lib/postgresql/18/docker \
#     -v $(pwd)/scratch:/scratch \
#     -e POSTGRES_USER=postgres \
#     -e POSTGRES_PASSWORD=abcd1234 \
#     postgres:18


docker run -d --name postgres_bbot \
    --rm \
    -p 5432:5432 \
    -v $(pwd)/data:/var/lib/postgresql/18/docker \
    -v $(pwd)/scratch:/scratch \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=abcd1234 \
    pgvector/pgvector:pg18
