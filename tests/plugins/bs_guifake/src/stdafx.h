/**
* \file stdafx.h
* \brief precompiled header
* \author Nikonov Max
* */
#ifndef BS_PRECOMPILED_HEADERS_H_
#define BS_PRECOMPILED_HEADERS_H_

#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <list>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <iterator>
#include <sstream>
#include <cmath>

#include <memory.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

//#include <boost/thread/condition.hpp>
//#include <boost/thread/mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/variant.hpp>
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/array.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/spirit.hpp>
#include <boost/spirit/core.hpp>
#include <boost/spirit/iterator/file_iterator.hpp>
#include <boost/date_time.hpp>
#include <boost/noncopyable.hpp>
#include <boost/type_traits.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/static_assert.hpp>

// Handles broken standard libraries better than <iterator>
#include <boost/detail/iterator.hpp>
#include <boost/throw_exception.hpp>
// FIXES for broken compilers
#include <boost/config.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif // #ifdef _OPENMP

#pragma intrinsic (memset, memcpy)

#include "smart_ptr.h"
#include "bs_common.h"
#include "bs_kernel.h"
#include "bs_link.h"
#include "bs_object_base.h"
#include "bs_tree.h"
#include "bs_exception.h"
#include "bs_prop_base.h"

#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/wrapper.hpp>
#include <boost/python/iterator.hpp>

#include "bs_plugin_common.h"
#include "py_bs_object_base.h"
#include "py_bs_command.h"
#include "py_bs_tree.h"
#endif

#ifdef _HDF5
#include "H5Cpp.h"
#endif // #ifdef _HDF5

#ifdef _MPI
#include "mpi_type_t.h"
#include "mpi_vector.h"
#endif  // #ifdef _MPI

#endif  // #ifndef BS_PRECOMPILED_HEADERS_H_
