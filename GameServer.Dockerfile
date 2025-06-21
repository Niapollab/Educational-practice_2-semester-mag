FROM python:3.6.8

COPY 'edupra-game-server/' '/usr/local/lib/python3.6/site-packages/edupra-game-server/'

RUN pip install --upgrade pip && \
    pip install -e '/usr/local/lib/python3.6/site-packages/edupra-game-server/'

ENTRYPOINT [ "edupra-game" ]
