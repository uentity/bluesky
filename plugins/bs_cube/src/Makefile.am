ACLOCAL_AMFLAGS = -I config/m4
plugins_LTLIBRARIES = libbs_cube.la
libbs_cube_la_SOURCES = bs_cube.cpp bs_cube_t.cpp main.cpp	
pluginsinc_HEADERS = bs_cube.h bs_cube_t.h
libbs_cube_la_LIBADD = -L../../../kernel/src -lloki -lblue-sky 
libbs_cube_la_CPPFLAGS = -DUNIX -DBS_EXPORTING_PLUGIN -DBSPY_EXPORTING_PLUGIN -I@top_srcdir@/kernel/src -I@top_srcdir@/kernel/src/python -I/usr/include/python2.5 -fvisibility=hidden -fvisibility-inlines-hidden
libbs_cube_la_CXXFLAGS = $(BOOST_CPPFLAGS) 
