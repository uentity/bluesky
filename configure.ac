AC_INIT([blue_sky], 0.1, [bos@ufanipi.ru])
AC_PREREQ([2.60])
AC_CONFIG_AUX_DIR(config)
AM_INIT_AUTOMAKE([-Wall 0.1 foreign dist-bzip2])
AC_DISABLE_STATIC

AC_PROG_CC
AC_PROG_CXX
AC_PROG_LIBTOOL


AX_PYTHON
AC_PYTHON_DEVEL(>='2.5')
AX_BOOST_BASE([1.33])
AX_BOOST_PYTHON
AX_BOOST_FILESYSTEM
AX_BOOST_IOSTREAMS
AX_BOOST_PROGRAM_OPTIONS
AX_BOOST_REGEX
AX_BOOST_SERIALIZATION
AX_BOOST_SIGNALS
AX_BOOST_SYSTEM
AX_BOOST_THREAD
PKG_CHECK_MODULES([LIBCONFIGPP], [libconfig++ >= 1.1.2],,exit)

data_suffix=$PACKAGE_NAME
AC_DEFINE_UNQUOTED([DATA_SUFFIX], ["$plugins"],[plugins dir])
AC_SUBST([pluginsdir], [$datadir/$data_suffix/plugins])
AC_SUBST([pluginsincdir],[$includedir/$data_suffix/plugins])
AC_SUBST([kernelincdir],[$includedir/$data_suffix/])
AC_SUBST([testdir],[$datadir/$data_suffix/tests])
#AC_SUBST([main_clientdir],[$datadir/$data_suffix/client])
#AC_SUBST([])

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile ./kernel/src/Makefile ./tests/plugins/Makefile \
                 ./plugins/bs_cube/src/Makefile \
		 ./tests/plugins/bs_guifake/src/Makefile \
		 ./tests/plugins/bs_gcube/src/Makefile \
                 ./python/bspy_loader/src/Makefile
                 ])
AC_OUTPUT

