ACLOCAL_AMFLAGS = -I config/m4
plugins_LTLIBRARIES = libbs_gcube.la

libbs_gcube_la_SOURCES = \
bs_gcube.cpp \
main.cpp

pluginsinc_HEADERS = \
bs_gcube.h

libbs_gcube_la_LIBADD = -L../../../../tests/plugins/bs_guifake/src -L../../../../kernel/src -lloki -lblue-sky -lbs_guifake
libbs_gcube_la_CPPFLAGS = -DUNIX -DBS_EXPORTING_PLUGIN -DBSPY_EXPORTING_PLUGIN -I@top_srcdir@/kernel/src -I@top_srcdir@/kernel/src/python -I@top_srcdir@/plugins/bs_cube/src -I@top_srcdir@/tests/plugins/bs_guifake/src -I/usr/include/python2.5 -fvisibility=hidden -fvisibility-inlines-hidden
libbs_gcube_la_CXXFLAGS = $(BOOST_CPPFLAGS)
