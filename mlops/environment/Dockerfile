FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:latest

USER dockeruser

# Conda is already installed
ENV CONDA_ENV_DIR=/opt/miniconda/envs

# Update environment variables
ENV AZUREML_CONDA_ENVIRONMENT_PATH="$CONDA_ENV_DIR/env"
ENV PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH"
ENV LD_LIBRARY_PATH="$AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH"

COPY conda.yml /tmp/

RUN conda env create --file /tmp/conda.yml && \
    conda run --prefix $AZUREML_CONDA_ENVIRONMENT_PATH pip cache purge && \
    conda clean --all --yes
