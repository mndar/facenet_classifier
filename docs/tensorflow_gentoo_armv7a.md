Cross Compiling Tensorflow Core for Gentoo armv7a on Gentoo amd64
-------
Date: 11 Feb 2018

Install Gentoo on your armv7a board and on your amd64 workstation as per instrutions in the Gentoo Handbook.

Then, on your workstation

1. **emerge crossdev**

2. **crossdev -S --gcc 5.4.0-r4 -t armv7a-hardfloat-linux-gnueabi**
Match the gcc version  with the one installed on your armv7a board. The current stage3 tarball (stage3-armv7a_hardfp-20161129.tar.bz2) has gcc 4.9.4. You need to upgrade that. Do this with **emerge gcc** on the armv7a board. It takes a few hours to do this as it part of the process doesn't use distcc. The gcc 5.4.0-r4 packages for armv7a and armv6 can be downloaded from http://www.czarsoftech.com/opensource/downloads/gcc

3. Clone the tensorflow repository
**git clone https://github.com/tensorflow/tensorflow **
**cd tensorflow **
**git checkout r1.4**
OR
Downloaded the latest tarball **tensorflow-1.5.0.tar.gz ** and unpack it

4. Download dependencies using 
**tensorflow/contrib/makefile/download_dependencies.sh **

5. Open tensorflow/contrib/makefile/Makefile in an editor and change it as required. Specifically you need to modify **CORE_CC_ALL_SRCS ** and **CORE_CC_EXCLUDE_SRCS ** depending on what capabilities you need to perform inference on your Tensorflow model.
Here is my Makefile http://www.czarsoftech.com/opensource/downloads/tensorflow/Makefile
I removed **-DIS_SLIM_BUILD** and **$(ANDROID_TYPES)** from **CXXFLAGS**. (This significantly increases the size of the built binary)

6.  You need to install the same protobuf version for both host and target. The tensorflow-1.5.0.tar.gz tarball had version 3.4.0.
You need to uninstall and remove the headers for all protobuf versions installed else you'll end up with useless application binaries complaining about incomptible protobuf versions.
First install it on your host & for your host using the usual
**./configure
make -j9
make install**
Then install the same protobuf version for your target on your host using 
**CC=armv7a-hardfloat-linux-gnueabi-gcc CXX=armv7a-hardfloat-linux-gnueabi-g++ ./configure --host=x86_64-pc-linux-gnu --target=armv7a-hardfloat-linux-gnueabi --prefix=/usr/armv7a-hardfloat-linux-gnueabi/usr/
make -j9**
*Note: Towards the end of the compilation process you'll get an error about not being able to execute the newly build protoc in /usr/armv7a-hardfloat-linux-gnueabi as it is a armv7a binary and not an x86_64 one. The worked around it by just replacing it with the one on the x86_64 root.*
Run **make -j9** again and this time the build should succeed.
Then, do **make install**

6. You now need to compile nsync and set 2 environment variables. Do that using
export HOST_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh`
export TARGET_NSYNC_LIB=`tensorflow/contrib/makefile/compile_nsync.sh -a armv7`
For x86_64, leave out the "-a armv7"
7. I then built libtensorflow-core.a with
**make -j9 -f tensorflow/contrib/makefile/Makefile HOST_OS=LINUX TARGET=PI OPTFLAGS="-Os" CXX=armv7a-hardfloat-linux-gnueabi-g++**
For x86_64, I used Gentoo AMD64 with gcc 5.4.0-r4 
**CC=x86_64-pc-linux-gnu-gcc-5.4.0 CXX=x86_64-pc-linux-gnu-g++-5.4.0 CFLAGS="-march=native" make -j9 -f tensorflow/contrib/makefile/Makefile HOST_OS=LINUX TARGET=LINUX OPTFLAGS="-Os"**
You can try *OPTFLAGS="-Os -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize"* but that results in a 'Bus Error' on my Rpi3 with Facenet, so had to remove it

8. While compiling your application, make sure you use --whole-archive else you'll end up with 'No Session Factory' error. Your LDFLAGS should look something like
**LDFLAGS += -L../lib -Wl,--allow-multiple-definition -Wl,--whole-archive -ltensorflow-core -lpthread -ldl -lprotobuf -lprotobuf-lite -lnsync -lz `pkg-config --libs opencv`**

9. To convert libtensorflow-core.a to a shared library:
**g++ -Os -std=c++11 -shared -fPIC -o libtensorflow.so -Wl,--allow-multiple-definition -Wl,--whole-archive libtensorflow-core.a -L. -lprotobuf -lprotobuf-lite -lm -lpthread -lz -ldl -Wl,--no-whole-archive**
Adjust the paths as per your installation

**Notes:**
*Refernece: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/makefile/README.md*
