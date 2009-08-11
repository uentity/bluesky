// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

#ifndef _BS_PY_OBJECT_BASE_H
#define _BS_PY_OBJECT_BASE_H

#include "py_bs_link.h"
#include "bs_object_base.h"
//#include "py_bs_command.h"
#include "py_bs_log.h"
#include "bs_kernel.h"
#include "py_bs_iterator.h"
#include "py_bs_messaging.h"

#include <vector>

namespace blue_sky {
namespace python {

class BS_API py_holder_iterator
	: public std::iterator<
		std::bidirectional_iterator_tag,
		py_objbase, ptrdiff_t,
		py_objbase, py_objbase > {
		friend class py_objbase;

private:
	typedef bs_objinst_holder::const_iterator sp_objinst_iter;
	typedef sp_objinst_iter(*get_end)();

	sp_objinst_iter ins;
	get_end endi;

public:
	py_holder_iterator(const py_holder_iterator&);
	py_holder_iterator(const bs_objinst_holder::const_iterator&,get_end);
	~py_holder_iterator();

	reference operator*() const;

	pointer operator->() const;

	py_holder_iterator& operator++();
	py_holder_iterator operator++(int);

	py_holder_iterator& operator--();
	py_holder_iterator operator--(int);

	bool operator ==(const py_holder_iterator&) const;
	bool operator !=(const py_holder_iterator&) const;
	const py_holder_iterator &operator =(py_holder_iterator&);
};

class BS_API py_objbase : public py_bs_messaging {
	friend class py_bs_inode;
	friend class py_bs_link;
	friend class py_bs_node;
	friend class py_kernel;
	friend class py_bs_slot;
	friend class py_bs_messaging;
	friend class py_holder_iterator;
	friend class python_slot;
	friend class py_bs_log;

	friend bool operator == (const py_objbase& tl, const py_objbase& tr);
	friend bool operator != (const py_objbase& tl, const py_objbase& tr);
public:
	typedef std::vector< py_objbase > py_bs_objinst_holder;

	typedef py_holder_iterator iterator;

	virtual ~py_objbase();

	type_descriptor bs_resolve_type() const;

	void ptr();
	sp_obj get_sp() const;

	// get reference count
	ulong refs() const;

	py_bs_inode inode() const;

	bool fire_signal(int signal_code, const py_objbase& params) const;
	std::vector< int > get_signal_list() const;

	//void feedback(int signal_code);
	//void link(const python_slot&);

	static py_holder_iterator begin();
	static py_holder_iterator end();

	//helper functions for extracting smart pointers to inherited types
	template < class T >
	smart_ptr< T, true > get_spx() const {
		return smart_ptr< T, true >(get_sp(), bs_dynamic_cast ());
	}

	template < class T >
#ifndef BS_DISABLE_MT_LOCKS
	lsmart_ptr< smart_ptr< T, true > > get_lspx() const {
		return get_spx< T >().lock();
#else
	smart_ptr< smart_ptr< T, true > > get_lspx() const {
		return get_spx< T >();
#endif
	}

	//if you've made a wrapped_t typedef then you can use the following
	template< class py_wrapper >
	static smart_ptr< typename py_wrapper::wrapped_t, true > get_spx(const py_wrapper* pw) {
		return pw->get_spx< typename py_wrapper::wrapped_t >();
	}

	template< class py_wrapper >
	static lsmart_ptr< smart_ptr< typename py_wrapper::wrapped_t, true > > get_lspx(const py_wrapper* pw) {
		return pw->get_lspx< typename py_wrapper::wrapped_t >();
	}

protected:
	py_objbase(const sp_obj&);
	py_objbase(const type_descriptor& td);

	template< class T >
	py_objbase(const Loki::Type2Type< T > obj_type)
		: py_bs_messaging(NULL), inode_(NULL)
	{
		sp = BS_KERNEL.create_object(Loki::Type2Type< T >::original_type::bs_type());
		assert(sp);
		spmsg = sp;
		inode_.spinode = sp->inode();
	}

	sp_obj sp;
	py_bs_inode inode_;
};
typedef py_objbase::py_bs_objinst_holder py_bs_objinst_holder;

inline bool operator == (const py_objbase& tl, const py_objbase& tr) {
	 return (tl.sp.get() == tr.sp.get());
}

inline bool operator != (const py_objbase& tl, const py_objbase& tr) {
	 return (tl.sp.get() != tr.sp.get());
}

template <class T>
class pyo : /*public T,*/ public py_objbase {
public:
	pyo() : /*T(),*/py_objbase(give_kernel::Instance().create_object(T::bs_type())) {}

	~pyo() {
		//this->del_ref();
		//give_kernel::Instance().release_object(sp);
	}
	/*static void *operator new(size_t) {
		return give_kernel::Instance().create_object(T::bs_type());
	}*/

	/*static void operator delete(void *p) {
		give_kernel::Instance().release_object((objbase*)p);
		//return ((T*)p)->bs_free_this();
	}*/
};

}	//namespace blue_sky::python
}	//namespace blue_sky

#endif // _BS_PY_OBJECT_BASE_H
