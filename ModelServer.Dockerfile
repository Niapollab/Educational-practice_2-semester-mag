FROM python:3.6.8

COPY 'gym-backgammon' '/usr/local/lib/python3.6/site-packages/gym-backgammon'
COPY 'edupra-core' '/usr/local/lib/python3.6/site-packages/edupra-core'
COPY 'edupra-model-builder' '/usr/local/lib/python3.6/site-packages/edupra-model-builder'
COPY 'edupra-model-server' '/usr/local/lib/python3.6/site-packages/edupra-model-server'

RUN pip install --upgrade pip && \
    pip install -e '/usr/local/lib/python3.6/site-packages/gym-backgammon' && \
    pip install -e '/usr/local/lib/python3.6/site-packages/edupra-core' && \
    pip install -e '/usr/local/lib/python3.6/site-packages/edupra-model-builder' && \
    pip install -e '/usr/local/lib/python3.6/site-packages/edupra-model-server'

ENTRYPOINT [ "edupra-server" ]
