ACLOCAL_AMFLAGS = -I config/m4
main_client_PROGRAMS = main_client
main_client_SOURCES =	main_client.cpp

main_client_LDADD = -L@top_builddir@/kernel/src \
	-L@top_builddir@/plugins/bs_cube/src -lboost_thread-mt -lpython2.5 -lblue_sky -lbs_cube
main_client_CPPFLAGS = -DUNIX  -I@top_srcdir@/kernel/include -I@top_srcdir@/kernel/include/python
main_client_CXXFLAGS = $(BOOST_CPPFLAGS) 
