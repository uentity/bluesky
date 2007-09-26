#!/bin/bash
# This script configures blue-sky and makes it

autoreconf --install
./configure --with-boost=/opt/boost #--with-boost=/your/boost/path (without "boost_1_34_1" like)
#export BS_PYTHON_INCLUDE=-I/usr/include/python2.5 #/your/python/path (like /usr/lib/python2.5)
#export BS_LIBCONFIGPP_CFLAGS=`pkg-config libconfig++ --cflags`
#export BS_LIBCONFIGPP_LIBS=`pkg-config libconfig++ --libs`
#make clean && make