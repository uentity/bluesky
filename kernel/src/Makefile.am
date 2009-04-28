ACLOCAL_AMFLAGS = -I config/m4
lib_LTLIBRARIES = libblue-sky.la
libblue_sky_la_SOURCES = 			\
	smart_ptr.cpp \
	bs_command.cpp		 		\
	bs_common.cpp		 		\
	bs_exception.cpp 			\
	bs_kernel.cpp				\
	bs_kernel_ston.cpp 			\
	bs_link.cpp 				\
	bs_report.cpp 				\
	bs_messaging.cpp 			\
	bs_misc.cpp 				\
	bs_object_base.cpp			\
	bs_prop_base.cpp 			\
	bs_shell.cpp 				\
	bs_tree.cpp 				\
	bs_type_info.cpp			\
	main.cpp 				\
	thread_pool.cpp 			\
	python/bs_import_common.cpp 		\
	python/py_bs_abstract_storage.cpp 	\
	python/py_bs_command.cpp 		\
	python/py_bs_kernel.cpp 		\
	python/py_bs_link.cpp 			\
	python/py_bs_log.cpp 			\
	python/py_bs_typed.cpp	\
	python/py_bs_object_base.cpp 		\
	python/bs_python_import.cpp		\
	python/py_bs_messaging.cpp 		\
	python/py_bs_shell.cpp 			\
	python/py_bs_tree.cpp 			\
	bs_abstract_storage.cpp
#libblue_sky_la_SOURCES = @kernel_cpp@
kernelinc_HEADERS = 				\
	include/python/py_bs_abstract_storage.h	\
	include/python/py_bs_command.h	\
	include/python/py_bs_iterator.h	\
	include/python/py_bs_kernel.h	\
	include/python/py_bs_link.h	\
	include/python/py_bs_log.h	\
	include/python/py_bs_messaging.h	\
	include/python/py_bs_object_base.h	\
	include/python/py_bs_shell.h	\
	include/python/py_bs_tree.h	\
	include/bs_abstract_storage.h	\
	include/bs_command.h	\
	include/bs_common.h	\
	include/bs_conversion.h	\
	include/bs_exception.h	\
	include/bs_fwd.h	\
	include/bs_imessaging.h	\
	include/bs_kernel.h	\
	include/bs_link.h	\
	include/bs_messaging.h	\
	include/bs_misc.h	\
	include/bs_object_base.h	\
	include/bs_plugin_common.h	\
	include/bs_prop_base.h	\
	include/bs_refcounter.h	\
	include/bs_report.h	\
	include/bs_shell.h	\
	include/bs_tree.h	\
	include/bs_type_info.h	\
	include/force_plugin_import.h	\
	include/setup_common_api.h	\
	include/setup_plugin_api.h	\
	include/smart_ptr.h	\
	include/stop_plugin_import.h	\
	include/thread_pool.h	\
	include/type_descriptor.h
#LIBCONFIG_INCLUDE = -I$(LIBCONFIG_PATH)/include #`pkg-config libconfig++ --cflags`
#LIBCONFIG_LIBS = -L$(LIBCONFIG_PATH)/lib #`pkg-config libconfig++ --libs`

libblue_sky_la_LIBADD = $(BOOST_SIGNALS_LIB) $(BOOST_FILESYSTEM_LIB) \
	$(BOOST_IOSTREAMS_LIB) $(BOOST_LDFLAGS) $(BOOST_REGEX_LIB) $(BOOST_SERIALIZATION_LIB) \
	$(BOOST_THREAD_LIB) $(BOOST_PYTHON_LIB) $(LIBCONFIGPP_LIBS) -ldl -lloki  \
	$(PYTHON_LDFLAGS) $(PYTHON_EXTRA_LIBS) $(PYTHON_EXTRA_LDFLAGS) $(BOOST_PYTHON_LIB) -lboost_python
libblue_sky_la_CPPFLAGS = -W -Wall -DUNIX -DBS_EXPORTING -DBS_EXPORTING_PLUGIN -fvisibility=hidden -fvisibility-inlines-hidden -I@top_srcdir@/kernel/include -I@top_srcdir@/kernel/include/python
libblue_sky_la_CXXFLAGS = $(BOOST_CPPFLAGS) $(LIBCONFIGPP_CFLAGS) $(PYTHON_CPPFLAGS)
