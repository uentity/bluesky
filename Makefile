################################################################################
# Project Makefile for build release version by GNU C++/FORTRAN on i686
################################################################################

################################################################################
# Result name
KERNEL_NAME	= blue-sky
PY_KERNEL_NAME = pyneo
BS_P1_NAME = bs_cube
BS_P2_NAME = bs_pool
BS_P3_NAME = bs_prop_storage
BS_P4_NAME = bs_matrix
BS_P5_NAME = bs_table
#BS_P6_NAME = bs_python_cube
BS_P7_NAME = bs_array
CLIENT_NAME = client

KERNEL_LIB_NAME = lib$(KERNEL_NAME).so
PY_KERNEL_LIB_NAME = lib$(PY_KERNEL_NAME).so
BS_P1_LIB_NAME = lib$(BS_P1_NAME).so
BS_P2_LIB_NAME = lib$(BS_P2_NAME).so
BS_P3_LIB_NAME = lib$(BS_P3_NAME).so
BS_P4_LIB_NAME = lib$(BS_P4_NAME).so
BS_P5_LIB_NAME = lib$(BS_P5_NAME).so
#BS_P6_LIB_NAME = lib$(BS_P6_NAME).so
BS_P7_LIB_NAME = lib$(BS_P7_NAME).so
CLIENT_EXE_NAME = $(CLIENT_NAME).exe

################################################################################
# Directory layout

KERNEL_DIR 	= ./kernel
PLUGINS_DIR 	= ./plugins
CLIENT_DIR      = ./client
RESULT_DIR	= ./exe/debug
PLUGINS_RESULT_DIR = $(RESULT_DIR)/plugins
PYTHON_DIR	= ./python
PY_KERNEL_DIR	= $(PYTHON_DIR)/kernel
PY_PLUGINS_DIR	= $(PYTHON_DIR)/plugins
DOC_DIR 	= ./doc

KERNEL_SRC_DIR  = $(KERNEL_DIR)/src
PY_KERNEL_SRC_DIR  = $(PY_KERNEL_DIR)/src
BS_P1_SRC_DIR = $(PLUGINS_DIR)/$(BS_P1_NAME)/src
BS_P2_SRC_DIR = $(PLUGINS_DIR)/$(BS_P2_NAME)/src
BS_P3_SRC_DIR = $(PLUGINS_DIR)/$(BS_P3_NAME)/src
BS_P4_SRC_DIR = $(PLUGINS_DIR)/$(BS_P4_NAME)/src
BS_P5_SRC_DIR = $(PLUGINS_DIR)/$(BS_P5_NAME)/src
#BS_P6_SRC_DIR = $(PY_PLUGINS_DIR)/$(BS_P6_NAME)/src
BS_P7_SRC_DIR = $(PLUGINS_DIR)/$(BS_P7_NAME)/src
CLIENT_SRC_DIR	= $(CLIENT_DIR)/src

KERNEL_INCLUDE_DIR = -I$(KERNEL_SRC_DIR) -I/usr/include/python2.5
PY_KERNEL_INCLUDE_DIR = -I$(KERNEL_SRC_DIR) -I/usr/include/python2.5 -I$(PY_KERNEL_SRC_DIR)
BS_P1_INCLUDE_DIR = $(KERNEL_INCLUDE_DIR) -I$(BS_P1_SRC_DIR) -I/usr/include/python2.5 -I$(PY_KERNEL_SRC_DIR)
BS_P2_INCLUDE_DIR = $(KERNEL_INCLUDE_DIR) -I$(BS_P2_SRC_DIR) -I$(BS_P7_SRC_DIR) -I/usr/include/python2.5 -I$(PY_KERNEL_SRC_DIR)
BS_P3_INCLUDE_DIR = $(KERNEL_INCLUDE_DIR) -I$(BS_P3_SRC_DIR)
BS_P4_INCLUDE_DIR = $(KERNEL_INCLUDE_DIR) -I$(BS_P4_SRC_DIR)
BS_P5_INCLUDE_DIR = $(KERNEL_INCLUDE_DIR) -I$(BS_P5_SRC_DIR) -I/usr/include/python2.5 -I$(PY_KERNEL_SRC_DIR)
#BS_P6_INCLUDE_DIR = $(PY_KERNEL_INCLUDE_DIR) -I$(BS_P6_SRC_DIR)
BS_P7_INCLUDE_DIR = $(KERNEL_INCLUDE_DIR) -I$(BS_P7_SRC_DIR)
CLIENT_INCLUDE_DIR  = $(BS_P1_INCLUDE_DIR) -I$(BS_P3_SRC_DIR) -I$(BS_P2_SRC_DIR) -I$(CLIENT_SRC_DIR) -I/usr/include/python2.5 -I$(PY_KERNEL_SRC_DIR)

KERNEL_BUILD_DIR  = $(KERNEL_DIR)/build
PY_KERNEL_BUILD_DIR  = $(PY_KERNEL_DIR)/build
BS_P1_BUILD_DIR = $(PLUGINS_DIR)/$(BS_P1_NAME)/build
BS_P2_BUILD_DIR = $(PLUGINS_DIR)/$(BS_P2_NAME)/build
BS_P3_BUILD_DIR = $(PLUGINS_DIR)/$(BS_P3_NAME)/build
BS_P4_BUILD_DIR = $(PLUGINS_DIR)/$(BS_P4_NAME)/build
BS_P5_BUILD_DIR = $(PLUGINS_DIR)/$(BS_P5_NAME)/build
#BS_P6_BUILD_DIR = $(PY_PLUGINS_DIR)/$(BS_P6_NAME)/build
BS_P7_BUILD_DIR = $(PLUGINS_DIR)/$(BS_P7_NAME)/build
CLIENT_BUILD_DIR  = $(CLIENT_DIR)/build

################################################################################
# Compiler flags layout

# Debug on switch
DEBUG		 = -g3
# Debug and profiling
#DEBUG		= -g -pg

#ifeq ($(findstring -pg, $(DEBUG)),)
# Not profiling
OPTIMIZE_COMMON   = -O3 -ffast-math -funroll-loops -finline-limit=1000000 #-fomit-frame-pointer
#OPTIMAZE_COMMON   =
#else
# Profiling
#OPTIMAZE_COMMON   = -O3 -ffast-math -funroll-loops -finline-limit=1000000
#endif

# Processor specific optimization (safe for debugging)
# This options should work on all Pentium's
#OPTIMAZE_SPECIFIC = -march=i686 -malign-double
# This options should work only on Pentium4
OPTIMIZE_SPECIFIC = #-march=pentium4 -msse2 -mfpmath=sse -malign-double

# Optimization options
OPTIMIZE	= -pipe $(OPTIMAZE_COMMON) $(OPTIMAZE_SPECIFIC)

# Warnings level
ifeq ($(findstring -O, $(OPTIMAZE)),)
WARNINGS_C	= -W -Wall -Wunused
else
WARNINGS_C	= -W -Wall -Wuninitialized -Wunused
endif

# Compiler
CC		= g++
# Linker
LINK		= g++

# Compiler flags
CFLAGS_COMMON	= -c -fpic -fvisibility=hidden -fvisibility-inlines-hidden -fno-check-new -mtune=core2 -pthread -MMD $(DEBUG) $(OPTIMIZE) $(WARNINGS_C) \
		-I$(BOOST_DIR)/include/boost-1_34 -I$(LOKI_DIR)/include -I$(LIBCONFIG_DIR)/include -DUNIX #-DUSING_LOKI_TYPEINFO
CFLAGS_KERNEL	= $(CFLAGS_COMMON) $(KERNEL_INCLUDE_DIR) -DBS_EXPORTING -DOUTPUT_DATETIME_MESSAGE
CFLAGS_PY_KERNEL = $(CFLAGS_COMMON) $(PY_KERNEL_INCLUDE_DIR) -D_PYTHON_EXPORT_ -D__PYTHON_ -DBS_EXPORTING
CFLAGS_BS_P1	= $(CFLAGS_COMMON) $(BS_P1_INCLUDE_DIR) -DBS_EXPORTING_PLUGIN -D__PYTHON_
CFLAGS_BS_P2	= $(CFLAGS_COMMON) $(BS_P2_INCLUDE_DIR) -DBS_EXPORTING_PLUGIN
CFLAGS_BS_P3	= $(CFLAGS_COMMON) $(BS_P3_INCLUDE_DIR) -DBS_EXPORTING_PLUGIN
CFLAGS_BS_P4	= $(CFLAGS_COMMON) $(BS_P4_INCLUDE_DIR) -DBS_EXPORTING_PLUGIN
CFLAGS_BS_P5	= $(CFLAGS_COMMON) $(BS_P5_INCLUDE_DIR) -DBS_EXPORTING_PLUGIN -D__PYTHON_
#CFLAGS_BS_P6	= $(CFLAGS_COMMON) $(BS_P6_INCLUDE_DIR) -D_PYTHON_EXPORT_PLUGIN_ -D__PYTHON_
CFLAGS_BS_P7	= $(CFLAGS_COMMON) $(BS_P7_INCLUDE_DIR) -DBS_EXPORTING_PLUGIN

CFLAGS_CLIENT   = -c -fpic -fno-check-new -march=i686 -pthread $(DEBUG) $(OPTIMIZE) $(WARNINGS_C) $(CFLAGS_COMMON) $(CLIENT_INCLUDE_DIR) -I/usr/include/python2.5
		#-DUNIX -DUSING_LOKI_TYPEINFO

# Linker flags
LDFLAGS 	= $(DEBUG) -shared $(OPTIMIZE_SPECIFIC)
LDFLAGS_CLIENT = $(DEBUG) $(OPTIMIZE_SPECIFIC)

# Linker additional libraries
#LIB_COMMON	= -L$(LOKI_SRC_DIR)/../lib -L$(BOOST_DIR)/lib -lloki
LIB_COMMON	= -L$(BOOST_DIR)/lib -L$(LOKI_DIR)/lib -L$(LIBCONFIG_DIR)/lib -lloki
LIB_KERNEL	= $(LIB_COMMON) -ldl -lboost_filesystem-gcc41-mt-d -lboost_thread-gcc41-mt-d -lboost_regex-gcc41-mt-d -lconfig++ -lboost_python-gcc41-mt-d
LIB_PY_KERNEL	= -L/usr/lib/python2.5/config -lpython2.5 -L$(BOOST_DIR)/lib -ldl -lboost_python-gcc41-mt-d -L$(RESULT_DIR) -lblue-sky
LIB_BS_P1	= $(LIB_COMMON) -L$(RESULT_DIR) -lpyneo #-lblue-sky -lpyneo #-l$(KERNEL_NAME)
LIB_BS_P2	= $(LIB_COMMON) -lboost_signals-gcc41-mt-d -lboost_python-gcc41-mt-d -L$(RESULT_DIR) -lpyneo
LIB_BS_P3	= $(LIB_COMMON) -L$(RESULT_DIR) -lpyneo
LIB_BS_P4	= $(LIB_COMMON) -L$(RESULT_DIR) -lpyneo
LIB_BS_P5	= $(LIB_COMMON) -L$(RESULT_DIR) -lpyneo
#LIB_BS_P6	= $(LIB_PY_KERNEL) -L$(PLUGINS_RESULT_DIR) -l$(BS_P1_NAME) -L$(RESULT_DIR) -lpyneo
LIB_BS_P7	= $(LIB_COMMON) -L$(RESULT_DIR) -lpyneo
LIB_CLIENT 	= $(LIB_COMMON) -L$(RESULT_DIR) -L$(PLUGINS_RESULT_DIR) -l$(KERNEL_NAME) -l$(BS_P1_NAME) -L/usr/lib/python2.5/config -lpython2.5 -L$(LOKI_DIR)/lib -lloki -lboost_signals-gcc41-mt-d
#LIB_CLIENT 	= $(LIB_COMMON) -L$(RESULT_DIR) -l$(KERNEL_NAME) -L$(LOKI_DIR)/lib -lloki -lboost_signals-gcc41-mt-d -L/usr/lib/python2.5/config -lpython2.5 -L$(BOOST_DIR)/lib -lboost_python-gcc41-mt-d #-L$(PLUGINS_RESULT_DIR) -l$(BS_P1_NAME)

################################################################################
# Commont project layout
################################################################################

KERNEL_RESULT = $(RESULT_DIR)/$(KERNEL_LIB_NAME)
PY_KERNEL_RESULT = $(RESULT_DIR)/$(PY_KERNEL_LIB_NAME)
BS_P1_RESULT = $(PLUGINS_RESULT_DIR)/$(BS_P1_LIB_NAME)
BS_P2_RESULT = $(PLUGINS_RESULT_DIR)/$(BS_P2_LIB_NAME)
BS_P3_RESULT = $(PLUGINS_RESULT_DIR)/$(BS_P3_LIB_NAME)
BS_P4_RESULT = $(PLUGINS_RESULT_DIR)/$(BS_P4_LIB_NAME)
BS_P5_RESULT = $(PLUGINS_RESULT_DIR)/$(BS_P5_LIB_NAME)
#BS_P6_RESULT = $(PLUGINS_RESULT_DIR)/$(BS_P6_LIB_NAME)
BS_P7_RESULT = #$(PLUGINS_RESULT_DIR)/$(BS_P7_LIB_NAME)
CLIENT_RESULT = $(RESULT_DIR)/$(CLIENT_EXE_NAME)

KERNEL_OBJS = \
	$(KERNEL_BUILD_DIR)/bs_common.o			\
	$(KERNEL_BUILD_DIR)/bs_command.o		\
	$(KERNEL_BUILD_DIR)/bs_type_info.o		\
	$(KERNEL_BUILD_DIR)/bs_object_base.o		\
	$(KERNEL_BUILD_DIR)/bs_exception.o		\
	$(KERNEL_BUILD_DIR)/bs_log.o			\
	$(KERNEL_BUILD_DIR)/bs_kernel.o			\
	$(KERNEL_BUILD_DIR)/bs_misc.o			\
	$(KERNEL_BUILD_DIR)/bs_prop_base.o		\
	$(KERNEL_BUILD_DIR)/bs_abstract_storage.o			\
	$(KERNEL_BUILD_DIR)/main.o

PY_KERNEL_OBJS = \
	$(PY_KERNEL_BUILD_DIR)/bs_python_kernel.o \
	$(PY_KERNEL_BUILD_DIR)/bs_py_object.o \
	$(PY_KERNEL_BUILD_DIR)/bs_python_object.o \
	$(PY_KERNEL_BUILD_DIR)/bs_python_command.o \
	$(PY_KERNEL_BUILD_DIR)/bs_import_common.o \
	$(PY_KERNEL_BUILD_DIR)/bs_python_import.o

BS_P1_OBJS = \
	$(BS_P1_BUILD_DIR)/bs_cube.o

BS_P2_OBJS = \
	$(BS_P2_BUILD_DIR)/bs_array_pool.o

BS_P3_OBJS = \
	$(BS_P3_BUILD_DIR)/storage_plugin.o

BS_P4_OBJS = \
	$(BS_P4_BUILD_DIR)/bs_mat_base.o		\
	$(BS_P4_BUILD_DIR)/bs_csr_matrix.o		\
	$(BS_P4_BUILD_DIR)/bs_diag_matrix.o		\
	$(BS_P4_BUILD_DIR)/main.o

BS_P5_OBJS = \
	$(BS_P5_BUILD_DIR)/bs_table.o

#BS_P6_OBJS = \
#	$(BS_P6_BUILD_DIR)/bs_python_cube.o

BS_P7_OBJS = \
	$(BS_P7_BUILD_DIR)/bs_array.o
	
CLIENT_OBJS = \
	$(CLIENT_BUILD_DIR)/main_client.o

DEPFILES = $(KERNEL_BUILD_DIR)/$(subst .o,.d,$(KERNEL_OBJS))		\
	   $(PY_KERNEL_BUILD_DIR)/$(subst .o,.d,$(PY_KERNEL_OBJS))	\
	   $(BS_P1_BUILD_DIR)/$(subst .o,.d,$(BS_P1_OBJS))		\
	   $(BS_P2_BUILD_DIR)/$(subst .o,.d,$(BS_P2_OBJS))		\
	   $(BS_P3_BUILD_DIR)/$(subst .o,.d,$(BS_P3_OBJS))		\
	   $(BS_P4_BUILD_DIR)/$(subst .o,.d,$(BS_P4_OBJS))		\
	   $(BS_P5_BUILD_DIR)/$(subst .o,.d,$(BS_P5_OBJS))		\
	   $(BS_P7_BUILD_DIR)/$(subst .o,.d,$(BS_P7_OBJS))		\
	   $(CLIENT_BUILD_DIR)/$(subst .o,.d,$(CLIENT_OBJS))

#	   $(BS_P6_BUILD_DIR)/$(subst .o,.d,$(BS_P6_OBJS))		\

################################################################################
# Build rules

all : $(KERNEL_RESULT) $(PY_KERNEL_RESULT) $(BS_P1_RESULT) $(BS_P2_RESULT) $(BS_P3_RESULT) \
	$(BS_P4_RESULT) $(BS_P5_RESULT) $(BS_P7_RESULT) $(CLIENT_RESULT)

$(KERNEL_RESULT) : $(KERNEL_OBJS)
	$(LINK) $(LDFLAGS) $(KERNEL_OBJS) $(LIB_KERNEL) -o $(KERNEL_RESULT)

$(PY_KERNEL_RESULT) : $(PY_KERNEL_OBJS)
	$(LINK) $(LDFLAGS) $(PY_KERNEL_OBJS) $(LIB_PY_KERNEL) -o $(PY_KERNEL_RESULT)

$(BS_P1_RESULT) : $(BS_P1_OBJS)
	$(LINK) $(LDFLAGS) $(BS_P1_OBJS) $(LIB_BS_P1) -o $(BS_P1_RESULT)

$(BS_P2_RESULT) : $(BS_P2_OBJS)
	$(LINK) $(LDFLAGS) $(BS_P2_OBJS) $(LIB_BS_P2) -o $(BS_P2_RESULT)

$(BS_P3_RESULT) : $(BS_P3_OBJS)
	$(LINK) $(LDFLAGS) $(BS_P3_OBJS) $(LIB_BS_P3) -o $(BS_P3_RESULT)

$(BS_P4_RESULT) : $(BS_P4_OBJS)
	$(LINK) $(LDFLAGS) $(BS_P4_OBJS) $(LIB_BS_P4) -o $(BS_P4_RESULT)

$(BS_P5_RESULT) : $(BS_P5_OBJS)
	$(LINK) $(LDFLAGS) $(BS_P5_OBJS) $(LIB_BS_P5) -o $(BS_P5_RESULT)

#$(BS_P6_RESULT) : $(BS_P6_OBJS)
#	$(LINK) $(LDFLAGS) $(BS_P6_OBJS) $(LIB_BS_P6) -o $(BS_P6_RESULT)

$(BS_P7_RESULT) : $(BS_P7_OBJS)
	$(LINK) $(LDFLAGS) $(BS_P7_OBJS) $(LIB_BS_P7) -o $(BS_P7_RESULT)

$(CLIENT_RESULT) : $(CLIENT_OBJS)
	$(LINK) $(LDFLAGS_CLIENT) $(CLIENT_OBJS) $(LIB_CLIENT) -o $(CLIENT_RESULT)

clean :
	rm -f $(KERNEL_RESULT) $(PY_KERNEL_RESULT) $(BS_P1_RESULT) $(BS_P2_RESULT) $(BS_P3_RESULT) 	\
		$(BS_P4_RESULT) $(BS_P5_RESULT) $(BS_P7_RESULT) $(CLIENT_RESULT) $(KERNEL_OBJS) 	\
		$(PY_KERNEL_OBJS) $(BS_P1_OBJS) $(BS_P2_OBJS) $(BS_P3_OBJS) $(BS_P4_OBJS) $(BS_P5_OBJS) \
		$(BS_P7_OBJS) $(CLIENT_OBJS) $(DEPFILES)

-include $(DEPFILES)

################################################################################
# Individual files build rules
$(BS_P1_BUILD_DIR)/%.o	: $(BS_P1_SRC_DIR)/%.cpp
	$(CC) $(CFLAGS_BS_P1) $< -o $@
	
$(BS_P2_BUILD_DIR)/%.o	: $(BS_P2_SRC_DIR)/%.cpp
	$(CC) $(CFLAGS_BS_P2) $< -o $@
	
$(BS_P3_BUILD_DIR)/%.o	: $(BS_P3_SRC_DIR)/%.cpp
	$(CC) $(CFLAGS_BS_P3) $< -o $@
	
$(BS_P4_BUILD_DIR)/%.o	: $(BS_P4_SRC_DIR)/%.cpp
	$(CC) $(CFLAGS_BS_P4) $< -o $@
	
$(BS_P5_BUILD_DIR)/%.o	: $(BS_P5_SRC_DIR)/%.cpp
	$(CC) $(CFLAGS_BS_P5) $< -o $@
	
#$(BS_P6_BUILD_DIR)/%.o	: $(BS_P6_SRC_DIR)/%.cpp
#	$(CC) $(CFLAGS_BS_P6) $< -o $@

$(BS_P7_BUILD_DIR)/%.o	: $(BS_P7_SRC_DIR)/%.cpp
	$(CC) $(CFLAGS_BS_P7) $< -o $@
	
$(KERNEL_BUILD_DIR)/%.o	: $(KERNEL_SRC_DIR)/%.cpp
	$(CC) $(CFLAGS_KERNEL) $< -o $@
	
$(PY_KERNEL_BUILD_DIR)/%.o	: $(PY_KERNEL_SRC_DIR)/%.cpp
	$(CC) $(CFLAGS_PY_KERNEL) $< -o $@
	
$(CLIENT_BUILD_DIR)/%.o	: $(CLIENT_SRC_DIR)/%.cpp
	$(CC) $(CFLAGS_CLIENT) $< -o $@
	