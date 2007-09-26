ACLOCAL_AMFLAGS = -I config/m4
lib_LTLIBRARIES = libblue-sky.la
libblue_sky_la_SOURCES = 			\
	python/py_bs_messaging.cpp 		\
	python/py_bs_shell.cpp 			\
	python/py_bs_tree.cpp 			\
	bs_abstract_storage.cpp	 		\
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
	python/bs_python_import.cpp
#libblue_sky_la_SOURCES = @kernel_cpp@
kernelinc_HEADERS = 				\
	python/py_bs_messaging.h 		\
	python/py_bs_shell.h 			\
	python/py_bs_tree.h 			\
	python/py_bs_exports.h	\
	bs_abstract_storage.h	 		\
	bs_command.h 				\
	bs_common.h				\
	bs_conversion.h 			\
	bs_exception.h 				\
	bs_fwd.h 				\
	bs_kernel.h			 	\
	bs_report.h 				\
	bs_messaging.h 				\
	bs_macro_private.h 			\
	bs_misc.h 				\
	bs_named_type.h 			\
	bs_object_base.h 			\
	bs_prop_base.h 				\
	bs_refcounter.h 			\
	bs_tree.h 				\
	bs_type_info.h 				\
	force_plugin_import.h 			\
	setup_common_api.h 			\
	setup_plugin_api.h 			\
	smart_ptr.h 				\
	smart_ptr.h			 	\
	stop_plugin_import.h 			\
	type_descriptor.h 			\
	bs_plugin_common.h 			\
	python/bs_import_common.h 		\
	python/py_bs_abstract_storage.h	 	\
	python/py_bs_command.h 			\
	python/py_bs_kernel.h 			\
	python/py_bs_link.h 			\
	python/py_bs_log.h 			\
	python/py_bs_object_base.h

#LIBCONFIG_INCLUDE = -I$(LIBCONFIG_PATH)/include #`pkg-config libconfig++ --cflags`
#LIBCONFIG_LIBS = -L$(LIBCONFIG_PATH)/lib #`pkg-config libconfig++ --libs`

libblue_sky_la_LIBADD = $(BOOST_SIGNALS_LIB) $(BOOST_FILESYSTEM_LIB) \
	$(BOOST_IOSTREAMS_LIB) $(BOOST_LDFLAGS) $(BOOST_REGEX_LIB) $(BOOST_SERIALIZATION_LIB) \
	$(BOOST_THREAD_LIB) $(BOOST_PYTHON_LIB) $(LIBCONFIGPP_LIBS) -ldl -lloki  \
	$(PYTHON_LDFLAGS) $(PYTHON_EXTRA_LIBS) $(PYTHON_EXTRA_LDFLAGS) $(BOOST_PYTHON_LIB) -lboost_python
libblue_sky_la_CPPFLAGS = -W -Wall -DUNIX -DBS_EXPORTING -DBS_EXPORTING_PLUGIN -fvisibility=hidden -fvisibility-inlines-hidden
libblue_sky_la_CXXFLAGS = $(BOOST_CPPFLAGS) $(LIBCONFIGPP_CFLAGS) $(PYTHON_CPPFLAGS)
