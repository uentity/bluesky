ACLOCAL_AMFLAGS = -I config/m4
plugins_LTLIBRARIES = libbs.la
libbs_la_SOURCES = main.cpp
libbs_la_LIBADD = -L../../../kernel/src -lloki -lblue-sky
libbs_la_CPPFLAGS =  -W -Wall -DUNIX -DBS_EXPORTING_PLUGIN -I@top_srcdir@/kernel/src -I@top_srcdir@/kernel/src/python -I/usr/include/python2.5 -fvisibility=hidden -fvisibility-inlines-hidden
libbs_la_CXXFLAGS = $(BOOST_CPPFLAGS)