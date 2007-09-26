#include "stdafx.h"

#include "base_gui.h"
#include "bs_report.h"
#include "bs_kernel.h"

namespace blue_sky {
	base_gui::base_gui(bs_type_ctor_param)
		: bs_refcounter()
		, objbase()
	{}

	base_gui::base_gui(const base_gui& src)
		: bs_refcounter()
		, objbase(src)
	{
		*this = src;
	}

	base_gui::~base_gui() {}

 	std::string base_gui::gfile(const type_descriptor &td) const {
		return "";
 	}

	BLUE_SKY_TYPE_STD_CREATE(base_gui)
	BLUE_SKY_TYPE_STD_COPY(base_gui)
	BLUE_SKY_TYPE_IMPL_SHORT(base_gui, objbase, "BS base GUI class")

	namespace python {
		py_base_gui::py_base_gui()
			: py_objbase(give_kernel::Instance().create_object(base_gui::bs_type()))
		{}

		py_base_gui::py_base_gui(sp_obj sp_obj_)
			: py_objbase(sp_obj_)
		{}

		py_base_gui::py_base_gui(const py_objbase &obj)
			: py_objbase(obj)
		{}

		py_base_gui::~py_base_gui() {}

		void py_base_gui::test () {}
	}
}
