#include "stdafx.h"

#include "bs_gui.h"
#include "bs_report.h"
#include "bs_kernel.h"

namespace blue_sky {
	bs_gui::bs_gui(bs_type_ctor_param)
		: bs_refcounter()
		, objbase()
	{
		BSOUT << "BS GUI starts..." << bs_end;
	}

	bs_gui::bs_gui(const bs_gui& src)
		: bs_refcounter()
		, objbase(src)
	{
		*this = src;
	}

	bs_gui::~bs_gui() {
		BSOUT << "BS GUI dies..." << bs_end;
	}

	void bs_gui::add_pair (const type_descriptor &cltd, const type_descriptor &guitd) {
		gui_holder.insert (std::pair <type_descriptor,type_descriptor> (cltd,guitd));
	}

	BLUE_SKY_TYPE_STD_CREATE(bs_gui)
	BLUE_SKY_TYPE_STD_COPY(bs_gui)
	BLUE_SKY_TYPE_IMPL_SHORT(bs_gui, objbase, "BS GUI main class")

	namespace python {
		py_bs_gui::py_bs_gui()
			: py_objbase(give_kernel::Instance().create_object(bs_gui::bs_type()))
		{}

		py_bs_gui::py_bs_gui(const py_objbase &obj)
			: py_objbase(obj)
		{}

		py_bs_gui::~py_bs_gui() {}

		void py_bs_gui::add_pair (const type_descriptor &cltd, const type_descriptor &guitd) {
			this->get_lspx (this)->add_pair (cltd,guitd);
		}
	}
}
