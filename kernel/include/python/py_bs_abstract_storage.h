/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _PY_BS_ABSTRACT_STORAGE_H
#define _PY_BS_ABSTRACT_STORAGE_H

#include "py_bs_object_base.h"
#include "bs_abstract_storage.h"

namespace blue_sky {
namespace python {

class BS_API py_bs_abstract_storage : public py_objbase	{
	 friend class py_kernel;
public:
	 py_bs_abstract_storage(const sp_storage&);

	 int open (const char *filename, int flags);
	 int close ();
	 bool is_opened () const;

	 int begin_object(const char *object_name);
	 int open_object(const char *object_name);

	 int end_object();

	 int write_int(const char *name, int value);
	 int write_float(const char *name, float value);
	 int write_double(const char *name, double value);
	 int write_char(const char *name, char value);
	 int write_void_int(const char *name, const void *value, int type);
	 int write_int_int_void(const char *name, int type, int size, const void* data);
	 int write_int_int_int_void(const char *name, int type, int rank, const int* dimensions, const void* data);

	 int read_val_type(const char *name, void *value, int type);
	 int read_type_data(const char *name, int type, void *data);

	 int get_rank(const char *name) const;

	 int get_dimensions(const char *name, int *dimensions) const;

private:
	 sp_storage spstor;
};

}	//end of namespace python
}	//end of namespace blue_sky

#endif // _PY_BS_ABSTRACT_STORAGE_H
