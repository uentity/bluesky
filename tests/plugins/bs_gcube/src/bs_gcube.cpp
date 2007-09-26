#include "stdafx.h"

#include "bs_gcube.h"
#include "bs_report.h"
#include "bs_kernel.h"

namespace blue_sky {
	bs_gcube::bs_gcube(bs_type_ctor_param param)
		: base_gui (param)
	{}

	bs_gcube::bs_gcube(const bs_gcube& src)
		: base_gui (src)
	{
		*this = src;
	}

	bs_gcube::~bs_gcube() {}

	std::string bs_gcube::gfile(const type_descriptor &td) const {
		return "/home/wolf/work/prog/blue_sky/plugins/bs_cube/python/bs_cube_gui.py";
	}

	BLUE_SKY_TYPE_STD_CREATE(bs_gcube)
	BLUE_SKY_TYPE_STD_COPY(bs_gcube)
	BLUE_SKY_TYPE_IMPL_SHORT(bs_gcube, base_gui, "BS cube GUI class")

	namespace python {
		py_bs_gcube::py_bs_gcube()
			: py_base_gui(give_kernel::Instance().create_object(bs_gcube::bs_type()))
		{}

		py_bs_gcube::py_bs_gcube(const py_objbase &obj)
			: py_base_gui(obj)
		{}

		py_bs_gcube::~py_bs_gcube() {}

		void py_bs_gcube::test () {}
	}
}
