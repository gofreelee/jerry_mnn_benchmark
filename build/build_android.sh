ABI="armeabi-v7a"
ANDROID_DIR=/data/local/tmp

cmake ../ \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI="${ABI}" \
        -DANDROID_STL=c++_static \
        -DCMAKE_BUILD_TYPE=Release \
        -DANDROID_NATIVE_API_LEVEL=android-21  \
        -DANDROID_TOOLCHAIN=clang++ \
        -DMNN_OPENCL:BOOL=ON \
        -DMNN_OPENMP:BOOL=ON \
        -DCMAKE_BUILD_TYPE=Debug \

make -j8

adb push run_test.out $ANDROID_DIR
adb shell "cd $ANDROID_DIR && source set_env.sh && ./run_test.out 0 3"