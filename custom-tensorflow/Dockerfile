# Use a base image with the desired Linux distribution and version
FROM ubuntu:18.04

# Update the package lists and install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    python3-dev \
    python3-pip \
    wget \
    unzip

# Set the working directory
WORKDIR /tensorflow_build

# Install Bazel
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.26.0/bazel-0.26.0-installer-linux-x86_64.sh && \
    chmod +x bazel-0.26.0-installer-linux-x86_64.sh && \
    ./bazel-0.26.0-installer-linux-x86_64.sh && \
    export PATH="$PATH:/usr/local/bin" && \
    bazel version

# Install other dependencies as needed
# For example, CUDA and cuDNN for GPU support

# Clone the TensorFlow repository
RUN git clone https://github.com/tensorflow/tensorflow.git /tensorflow_build/tensorflow &&\
    cd /tensorflow_build/tensorflow &&\
    git checkout r2.0 && \
    pip3 install -U --user pip six numpy wheel setuptools mock future>=0.17.1 && \
    pip3 install -U --user keras_applications==1.0.6 --no-deps && \
    pip3 install -U --user keras_preprocessing==1.0.5 --no-deps

# # Configure the build (customize as needed)
# RUN cd /tensorflow_build/tensorflow && ./configure

CMD ["bash"]

# # # Build TensorFlow
# # RUN bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

# # # Package TensorFlow as a pip package
# # RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# # # Install the built TensorFlow package
# # RUN pip3 install /tmp/tensorflow_pkg/tensorflow-version.whl

# # Set the default command when the Docker container is run
# CMD ["bash"]
