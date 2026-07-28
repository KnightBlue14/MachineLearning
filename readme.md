A collection of Machine Learning samples in Python. Mostly for myself, but free to anyone else who can use it.

UPDATE:
Since writing this, I have upgraded my PC to an AMD graphics card on a Linux Operating System, meaning that paths and directories don't quite work the same way. The recommended way to do this is to use a docker container from ROCM (AMD's equivalent to CUDA (Don't worry, nothing needs to be rewritten on that front)) hosting a Jupyter environment. While it does need to be set up and restarted each time the container is closed, it also means that everything needed is inside the container, and won't cause any conflicts with the rest of your system.

The only major differences I have found so far are that I need to enter terminal commands to install models in the Ollama container and run streamlit, as well as run instructions that I have included in AMD_instructions.txt to make sure the rocm container can interact with Ollama. Other than that, once everything has been re-installed, it all works just fine.

I've also added the .ipynb files to this repo, they are the same in terms of content, just better for a jupyter environment.