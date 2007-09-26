#!/bin/bash
# This script makes blue-sky

export BS_PYTHON_INCLUDE=-I/usr/include/python2.5 #/your/python/path (like /usr/lib/python2.5)
export BS_LIBCONFIGPP_CFLAGS=`pkg-config libconfig++ --cflags`
export BS_LIBCONFIGPP_LIBS=`pkg-config libconfig++ --libs`
make