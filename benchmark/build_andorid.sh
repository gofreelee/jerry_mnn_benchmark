set -e
ABI="armeabi-v7a"
OPENMP="ON"
VULKAN="ON"
OPENCL="ON"
OPENGL="OFF"
RUN_LOOP=10
FORWARD_TYPE=0
CLEAN=""
PUSH_MODEL=""

WORK_DIR=`pwd`
BUILD_DIR=build
BENCHMARK_MODEL_DIR=$WORK_DIR/models
ANDROID_DIR=/data/local/tmp

 if [ "-c" == "$CLEAN" ]; then
        clean_build $BUILD_DIR
fi
mkdir -p build
cd $BUILD_DIR
cmake ../../ \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DANDROID_ABI="${ABI}" \
        -DANDROID_STL=c++_static \
        -DCMAKE_BUILD_TYPE=Release \
        -DANDROID_NATIVE_API_LEVEL=android-21  \
        -DANDROID_TOOLCHAIN=clang \
        -DMNN_VULKAN:BOOL=$VULKAN \
        -DMNN_OPENCL:BOOL=$OPENCL \
        -DMNN_OPENMP:BOOL=$OPENMP \
        -DMNN_OPENGL:BOOL=$OPENGL \
        -DMNN_DEBUG:BOOL=OFF \
        -DMNN_BUILD_BENCHMARK:BOOL=ON \
        -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
        -DNATIVE_LIBRARY_OUTPUT=.
make -j8 benchmark.out timeProfile.out benchmarkExprModels.out

adb push benchmarkExprModels.out $ANDROID_DIR
adb shell "cd $ANDROID_DIR && source set_env.sh && ./benchmarkExprModels.out SqueezeNet_100 2 0 1"