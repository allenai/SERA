# Installation

Clone the repository locally, and then set up the environment.

With pip:
```
conda create -n sera python=3.12
pip install -e . -e modules/codeflow -e modules/SWE-agent 
pip install flash-attn==2.7.4.post1 --no-build-isolation # Only needed if using this package to train as well.
```

With uv:
```
uv pip install -e . -e modules/codeflow -e modules/SWE-agent
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

How to create own gh org, docker org

TODO: 
ask claude to make all top of file comments good
ask claude to set up uv