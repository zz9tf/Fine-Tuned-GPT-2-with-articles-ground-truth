cd /tensorflow_build/tensorflow

# Configure the build (customize as needed)
./configure

# Set your configuration for package tensorflow
# AVX                         --copt=-mavx
# AVX2                        --copt=-mavx2
# FMA                         --copt=-mfma
# SSE 4.1                     --copt=-msse4.1
# SSE 4.2                     --copt=-msse4.2
# All supported by processor  --copt=-march=native
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package