/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief BlueSky smart pointers machinery
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _SMART_PTR_H
#define _SMART_PTR_H

#include "bs_conversion.h"
#include "bs_refcounter.h"
#include "bs_imessaging.h"
#include "setup_common_api.h"

#include "boost/type_traits/remove_const.hpp"
#include "boost/type_traits/add_const.hpp"
#include "boost/checked_delete.hpp"
#include "boost/version.hpp"
//thread-safe boost's counter, used in shared_ptr, can utilize custom deleters - for smart_ptr< T, false >
#if BOOST_VERSION / 100000 == 1 && BOOST_VERSION / 100 % 1000 < 39
#include "boost/detail/shared_count.hpp"
#else
//#include "boost/smart_ptr/detail/shared_count.hpp"
// shared_count_132.hpp cause compile error without this header first
#include <boost/serialization/assume_abstract.hpp>
// use shared_count compatible with boost serialization (public sp_counted_base*)
#include <boost/serialization/detail/shared_count_132.hpp>
#endif
// provide access to private members for serialization machinery
#include <boost/serialization/access.hpp>

#include "loki/TypeManip.h"
//#include <iostream>

#ifdef BSPY_EXPORTING
#include <boost/python/pointee.hpp>
#endif

// if you SURELY don't need locks in mt environment
// then define the following key
// TODO: replace this with better solution
//#define BS_DISABLE_MT_LOCKS

/*!
* \brief
* Casting behavior used by smart pointers to blue-sky types
*
* Possible values:
*
* bs_static_cast - fully static check both up and down the class hierarchy during compile time.
* Consider class A is the base of class B with any inheritance level. Then the following low-level conversions are legal:
* A* = B* and B* = A* as well as all dependent higher level casts, i.e. smart_ptr< A > = smart_ptr< B >,
* smart_ptr< B >(smart_ptr< A >), etc. The resulting smart_ptr will definitevely be non-NULL if the source
* smart_ptr is non-NULL. On the other side, if A and B are on different branches of class hierarchy, then all casts
* between them become illegal and you will get a compile-time error when trying to do any such cast.
* All casts done using static_cast, so the maximum performance is reached. Limitations of this policy are:
* - Types A and B should be completely defined;
* - You should avoid situations like following:
* smart_ptr< A > p_a = new A;
* smart_ptr< B > p_b = p_a; //this will compile fine, p_b points to instance of A
* //unpredeicted behaviuor - runtime error in most cases
* //because p_b really points to instance of A, not B
* if(p_b) p_b->B_specific_member();
*
* bs_semi_dynamic_cast - static_cast used only when casting up the class hierarchy, dynamic_cast used when casting
* down (from parent to child).
* In the example above, A* = B* is "statically" legal, B* = A* is also legal but implemented through dynamic_cast.
* It means that the result of cast like smart_ptr< B > = smart_ptr< A > will depend on if smart_ptr< A > really points
* to instance of B. If so, smart_ptr< B > will be equal to smart_ptr< A > and NULL otherwise. As in
* previuos case, conversions between different hierarchy branches are illegal and will result in compile-time error.
* Our dangerous example now works fine:
* smart_ptr< B > p_b = p_a; //p_b = NULL, because p_a points to instance of A
* //NULL-check prevents from calling non-existent function
* if(p_b) p_b->B_specific_member();
* This policy is a tradeoff between performance and stability. Performance is degraded because of dynamic_casts,
* but now you can avoid all danger situations by NULL-checks. The only requirement is that types A and B should be
* completely defined.
* This policy is used by default, but you can override it by using special ctors or assgin() member function.
*
* bs_dynamic_cast - fully dynamic cast.
* Any conversions are legal and performed by dynamic_cast. No class hierarchy compile-time checks are made, so
* all casts will compile fine. But you should always check resulting smart_ptr for being non-NULL.
* This is a slowest policy, which can work with incomplete types.
*
* bs_implicit_cast - use compiler implicit casting
*/

#ifndef BS_DEF_CAST_POLICY
//#ifdef UNIX
//#define BS_DEF_CAST_POLICY bs_static_cast
//#else
#define BS_DEF_CAST_POLICY bs_semi_dynamic_cast
//#endif
#endif

#define BS_SP_PURE_POINTER_ACCESS                   \
public:                                             \
pure_pointer_t operator->() const {                 \
	return const_cast< pure_pointer_t >(this->p_);  \
}                                                   \
pure_ref_t operator*() const {                      \
	return *const_cast< pure_pointer_t >(this->p_); \
}                                                   \
operator pure_pointer_t() const {                   \
	return const_cast< pure_pointer_t >(this->p_);  \
}                                                   \
pure_pointer_t get() const {                        \
	return const_cast< pure_pointer_t >(this->p_);  \
}

namespace blue_sky {

void BS_API
bs_refcounter_add_ref(const bs_refcounter *p);

void BS_API
bs_refcounter_del_ref(const bs_refcounter *p);

enum bs_cast_policy {
	BS_IMPLICIT_CAST = 0,
	BS_STATIC_CAST = 1,
	BS_SEMI_DYNAMIC_CAST = 2,
	BS_DYNAMIC_CAST = 3
};

template< bs_cast_policy policy >
struct bs_castpol_val {
	enum { value = policy };
};

typedef bs_castpol_val< BS_STATIC_CAST > bs_static_cast;
typedef bs_castpol_val< BS_SEMI_DYNAMIC_CAST > bs_semi_dynamic_cast;
typedef bs_castpol_val< BS_DYNAMIC_CAST > bs_dynamic_cast;
typedef bs_castpol_val< BS_IMPLICIT_CAST > bs_implicit_cast;

/*!
\struct null_deleter
\ingroup smart_pointers
\brief Custom deleter - does nothing.

Taken from boost documentation.
*/
struct null_deleter_unsafe {
	void operator()(void *) const
	{}
};

struct BS_API usual_deleter_unsafe {
	void operator()(void *p) const;
};

struct BS_API array_deleter_unsafe {
	void operator()(void *p) const;
};

template< class T >
struct usual_deleter {
	void operator()(T* p) const {
		typedef char type_must_be_complete[ sizeof(T)? 1: -1 ];
		(void) sizeof(type_must_be_complete);
		delete p;
	}
};

template< class T >
struct array_deleter {
	void operator()(T* p) const {
		typedef char type_must_be_complete[ sizeof(T)? 1: -1 ];
		(void) sizeof(type_must_be_complete);
		delete []p;
	}
};

/*!
struct bs_obj_deleter< T >
\ingroup smart_pointers
\brief Deleter for blue-sky objects. Dereferences pointed object
*/
template< class T >
struct bs_obj_deleter {
	void operator()(T* p) const {
		static_cast< const bs_refcounter* >(p)->del_ref();
	}
};

struct BS_API bs_obj_deleter_unsafe {
	void operator()(void * p) const;
};

//! \namespace bs_private
namespace bs_private {

	/*!
	class smart_ptr_base< T >
	\ingroup smart_pointers
	\brief Base class for all smart pointers
	*/
	template< class T >
	class smart_ptr_base {
#if defined(_MSC_VER)
		friend class smart_ptr_base;
#else
		template< class R > friend class smart_ptr_base;
#endif

	public:
		typedef T pointed_t;
		typedef T* pointer_t;
		typedef T& ref_t;
		typedef smart_ptr_base< T > this_t;

		/*!
		\brief Default constructor from simple pointer.
		*/
		explicit smart_ptr_base(pointer_t lp = NULL) : p_(lp) {}


		/*!
		\brief Templated constructor from simple pointer using default casting policy.
		*/
		template< class R >
		smart_ptr_base(R* lp) {
			init(lp, cast_helper< R, BS_DEF_CAST_POLICY >::result_t());
		}

		/*!
		\brief Templated constructor from simple pointer using custom casting policy.
		\param lp - any simple pointer. Supplied casting policy is used to determine if R can be casted to T
		*/
		template< class R, bs_cast_policy castp_id >
		smart_ptr_base(R* lp, bs_castpol_val< castp_id >) {
			init(lp, cast_helper< R, bs_castpol_val< castp_id > >::result_t());
		}

		/*!
		\brief Templated constructor from smart_ptr_base using default casting policy
		*/
		template< class R >
		smart_ptr_base(const smart_ptr_base< R >& lp) {
			init(lp.p_, cast_helper< R, BS_DEF_CAST_POLICY >::result_t());
		}

		/*!
		\brief Templated constructor from smart_ptr_base using custom casting policy
		*/
		template< class R, bs_cast_policy castp_id >
		smart_ptr_base(const smart_ptr_base< R >& lp, bs_castpol_val< castp_id >) {
			init(lp.p_, cast_helper< R, bs_castpol_val< castp_id > >::result_t());
		}

		/*!
		\brief Copy constructor.
		Standard is fine
		*/
		//smart_ptr_base(const this_t& lp) : p_(lp.p_) {}

		// dtor
		virtual ~smart_ptr_base() {}

		/*!
		\brief Assignment from simple pointer of any type using custom casting policy
		*/
		template< bs_cast_policy castp_id, class R >
		this_t& assign(R* lp) {
			init(lp, cast_helper< R, bs_castpol_val< castp_id > >::result_t());
			return *this;
		}

		/*!
		\brief Assignment from smart_ptr_base of any blue-sky using custom casting policy
		*/
		template< bs_cast_policy castp_id, class R >
		this_t& assign(const smart_ptr_base< R >& lp) {
			init(lp.p_, cast_helper< R, bs_castpol_val< castp_id > >::result_t());
			return *this;
		}

		/*!
		\brief Access to inner simple pointer.
		*/
		pointer_t get() const { return p_; }
		//const pointed_t* get_ptr() const { return p_; }

		/*!
		\brief Conversion to simple pointer operator
		*/
		operator pointer_t() const { return p_; }
		//operator const pointed_t*() const { return p_; }

		/*!
		\brief Dereference operator.
		*/
		ref_t operator*() const { return *p_; }

		/*!
		\brief Member-access operator
		*/
		pointer_t operator->() const { return p_; }

		operator bool() const {
			return (p_ != NULL);
		}

	protected:
		//! inner simple pointer
		pointer_t p_;

		template< class R, class cast_pol >
		struct cast_helper {
			typedef conversion< R, pointed_t > c;
			enum { castp_val = cast_pol::value };
			enum { use_dynamic_cast = (castp_val == (int)BS_DYNAMIC_CAST)
				|| (castp_val == (int)BS_SEMI_DYNAMIC_CAST && !c::exists_uc && c::exists1of2way_uc) };
			enum { compile_error = (castp_val == (int)BS_STATIC_CAST || castp_val == (int)BS_SEMI_DYNAMIC_CAST)
				&& !c::exists1of2way_uc };
			//0 - use implicit cast, 1 - use static_cast, 2 - use dynamic_cast, 3 - emit compile-time error
			enum { result = (1 + use_dynamic_cast + 2*compile_error) & 3*(castp_val != (int)BS_IMPLICIT_CAST)};

			typedef Loki::Int2Type< result > result2type;
			//this function added to avoid 'typename' addition everywhere
			static result2type result_t() { return result2type(); }
		};

		/*!
		\brief Helper initialization function
		Uses implicit task rules offered by compiler
		*/
		template< class R >
		void init(R* lp, Loki::Int2Type< 0 >) {
			p_ = lp;
		}

		/*!
		\brief Helper initialization function
		Specialization for castable simple pointer.
		*/
		template< class R >
		void init(R* lp, Loki::Int2Type< 1 >) {
			p_ = static_cast< pointer_t >(lp);
		}

		template< class R >
		void init(R* lp, Loki::Int2Type< 2 >) {
			p_ = dynamic_cast< pointer_t >(lp);
		}

		/*!
		\brief Helper initialization function
		Specialization for non-castable simple pointer. Should produce a compile-time error
		*/
		template< class R >
		void init(R* lp, Loki::Int2Type< 3 >) {
			struct R_cannot_be_casted_to_T;
			R_cannot_be_casted_to_T error;
		}
	};

	/*!
	class smart_ptr_base< void >
	\ingroup smart_pointers
	\brief Explicit specialization for void - no dereferencing operator.
	*/
	template< >
	class smart_ptr_base< void > {
	public:
		typedef void pointed_t;
		typedef void* pointer_t;
		typedef void ref_t;
		typedef smart_ptr_base< void > this_t;

		explicit smart_ptr_base(void* lp = NULL) : p_(lp) {}
		smart_ptr_base(const smart_ptr_base<void>& lp) : p_(lp.p_) {}

		//! everything can be converted to void* so always using implicit conversion rules
		template< class R >
		smart_ptr_base(R* lp)
			: p_(lp)
		{}

		template< class R, class cast_t >
		smart_ptr_base(R* lp, cast_t cast)
			: p_(lp)
		{}

		template< class R >
		smart_ptr_base(const smart_ptr_base< R >& lp)
			: p_(lp.p_)
		{}

		template< class R, class cast_t >
		smart_ptr_base(const smart_ptr_base< R >& lp, cast_t cast)
			: p_(lp.p_)
		{}

		// dtor
		virtual ~smart_ptr_base() {}

		template< class cast_t, class R >
		this_t& assign(R* lp) {
			p_ = lp;
			return *this;
		}

		template< class cast_t, class R >
		this_t& assign(const smart_ptr_base< R >& lp) {
			p_ = lp.p_;
			return *this;
		}

		pointer_t get() const { return p_; }
		//const void* get_ptr() const { return p_; }

		operator pointer_t() const { return p_; }
		//operator const void*() const {return p_; }

		pointer_t operator->() const { return p_; }

	protected:
		//! inner simple pointer
		pointer_t p_;
	};

	struct deleter_base {
		// null deleter policy used by default
		virtual void dispose(void const*) const {
			delete this;
		};

		// virtual dtor -- assume that deleter created with new
		virtual ~deleter_base() {}
	};

	template< class D >
	struct deleter_adaptor : public deleter_base {
		deleter_adaptor(const D& d = D()) : d_(d) {}

		void dispose(void const* p) const {
			d_(p);
			delete this;
		}

	private:
		const D d_;
	};

	// refcounter around simple ptr - delete holded ptr on dispose
	template< class T, class base_refcnt = bs_refcounter >
	class bs_refcounter_sp : public base_refcnt {
	public:
		typedef T* pointer_t;

		explicit bs_refcounter_sp(pointer_t p)
			: base_refcnt(1), p_(p)
		{}

		template< class R >
		explicit bs_refcounter_sp(R* p)
			: base_refcnt(1), p_(p)
		{}

		void dispose() const {
			usual_deleter< T >()(p_);
			delete this;
		}

		pointer_t get() const {
			return p_;
		}

	private:
		friend class boost::serialization::access;

		// special ctor for restoring after serialization with refcount = 0
		explicit bs_refcounter_sp(pointer_t p, int)
			: base_refcnt(0), p_(p)
		{}

		pointer_t p_;
	};

	// same as above, but with custom deleter
	template< class T, class D, class base_refcnt = bs_refcounter >
	class bs_refcounter_spd : public base_refcnt {
	public:
		typedef T* pointer_t;

		explicit bs_refcounter_spd(pointer_t p)
			: base_refcnt(1), p_(p) //, d_(d)
		{}

		template< class R >
		explicit bs_refcounter_spd(R* p)
			: base_refcnt(1), p_(p)
		{}

		void dispose() const {
			//d_(p_);
			D()(p_);
			delete this;
		}

		pointer_t get() const {
			 return p_;
		}

	private:
		friend class boost::serialization::access;

		// special ctor for restoring after serialization with refcount = 0
		explicit bs_refcounter_spd(pointer_t p, int)
			: base_refcnt(0), p_(p) //, d_(d)
		{}

		pointer_t p_;
		//D d_;
	};

	// refcounter around simple ptr - auto inc ref in copy ctor, dec ref in dtor
	// can be used like boost::shared_count
	template< class T, class base_refcnt = bs_refcounter >
	class bs_refcounter_ptr {
	public:
		typedef T* pointer_t;

		explicit bs_refcounter_ptr(pointer_t p)
			: rc_(new bs_refcounter_sp< T, base_refcnt >(p))
		{}

		// second param only for compatibility with boost::detail::shared_count
		template< class D >
		explicit bs_refcounter_ptr(pointer_t p, D)
			: rc_(new bs_refcounter_spd< T, D, base_refcnt >(p))
		{}

		template< class R >
		explicit bs_refcounter_ptr(R* p)
			: rc_(new bs_refcounter_sp< T, base_refcnt >(p))
		{}

		template< class R, class D >
		explicit bs_refcounter_ptr(R* p, D)
			: rc_(new bs_refcounter_spd< T, D, base_refcnt >(p))
		{}

		bs_refcounter_ptr(const bs_refcounter_ptr& lhs)
			: rc_(lhs.rc_)
		{
			if(rc_)
				rc_->add_ref();
		}

		template< class R >
		bs_refcounter_ptr(const bs_refcounter_ptr< R, base_refcnt >& lhs)
			: rc_(lhs.rc_)
		{
			if(rc_)
				rc_->add_ref();
		}

		// dtor
		~bs_refcounter_ptr() {
			if(rc_)
				rc_->del_ref();
		}

		void swap(bs_refcounter_ptr& lhs) {
			std::swap(rc_, lhs.rc_);
		}

		bs_refcounter_ptr& operator=(const bs_refcounter_ptr& lhs) {
			if(rc_ != lhs.rc_)
				bs_refcounter_ptr(lhs).swap(*this);
			return *this;
		}

		bs_refcounter* operator->() const {
			return rc_;
		}

		long use_count() const {
			return rc_ != 0 ? rc_->refs() : 0;
		}

		friend inline bool operator==(const bs_refcounter_ptr& a, const bs_refcounter_ptr& b) {
			return a.rc_ == b.rc_;
		}

	private:
		friend class boost::serialization::access;
#if defined(_MSC_VER)
		friend class bs_refcounter_ptr;
#else
		template< class R, class BR > friend class bs_refcounter_ptr;
#endif

		// empty ctor for serialization purposes
		bs_refcounter_ptr() : rc_(NULL) {}

		// pointer to reference counter
		base_refcnt* rc_;
	};

	//function that fires unlock signal
	template< class T, bool can_fire = conversion< T, bs_imessaging >::exists_uc >
	struct signal_unlock;
} // end of namespace bs_private

/*!
class bs_locker< T >
\ingroup smart_pointers
\brief Proxy locker object. Automatically locks pointed object in constructor and unlocks in desctructor

Looks like smart pointer from calling side (contains appropriate members)
Used for quick non-constant members access from smart pointers. Designed to be created and destroyed automatically
on stack by compiler.
For manual object locking use lsmart_ptr constructed from smart_ptr.
*/
template< class T >
class bs_locker : public bs_private::smart_ptr_base< typename boost::remove_const< T >::type > {
	typedef bs_locker< T > this_t;
	typedef typename boost::remove_const< T >::type pointed_t;
	typedef typename boost::add_const< T >::type const_pointed_t;
	typedef bs_private::smart_ptr_base< pointed_t > base_t;

	typedef pointed_t* pointer_t;
	typedef const_pointed_t* const_pointer_t;
	typedef pointed_t& ref_t;
	typedef const_pointed_t& const_ref_t;

	using base_t::p_;

public:
	/*!
	\brief Constructor from simple pointer. Locks pointed object

	Made public in order to be used by objbase.
	\param lp - pointer to const object
	*/
	bs_locker(const_pointer_t lp)
		: base_t(const_cast< pointer_t >(lp))
		  , lobj_(lp->mutex())
	{}
	/*!
	\brief Constructor from simple pointer with disjoint mutex.
	*/
	bs_locker(const_pointer_t lp, bs_mutex& m)
		: base_t(const_cast< pointer_t >(lp)), lobj_(m)
	{}

	operator ref_t() const {
		return (*p_);
	}

	/*!
	\brief Assignment operator from reference to object

	Allows transparent assignment like 'ptr.lock() = a'
	\param r - reference to const object
	\return nonconst reference to pointed type.
	*/
	ref_t operator =(const_ref_t r) const {
		*p_ = r;
		return *p_;
	}

	//dtor
	~bs_locker() {
		//fire unlock signal
		//bs_private::signal_unlock< pointed_t >::fire(p_);
	}

	/*!
	\brief Dumb copy construction

	Just to let some code compile. This is really never used.
	*/
	bs_locker(const bs_locker&);

private:
	//! type of locker object used in mutex
	typename bs_mutex::scoped_lock lobj_;

	/*!
	\brief Empty constructor prevents from creating empty object.
	*/
	bs_locker();
	//prevent copying
	//bs_locker(const bs_locker&);

	/*!
	\brief Prevent assignment of locker objects
	*/
	const bs_locker& operator=(const bs_locker&);
};

//----------------------------------------------------------------------------------------------------
/*!
class lsmart_ptr < SP >
\ingroup smart_pointers
\brief Locked smart pointer. Pointed object is locked during lifetime of lsmart_ptr
Unlock is done automatically in destructor
*/
template< class SP >
class lsmart_ptr : public SP
{
	//typedefs
public:
	typedef SP base_t;
	typedef lsmart_ptr< SP > this_t;
	typedef typename base_t::pointed_t pointed_t;
	typedef typename base_t::pointer_t pointer_t;
	typedef typename base_t::ref_t ref_t;
	typedef typename base_t::pure_pointed_t pure_pointed_t;
	typedef typename base_t::pure_pointer_t pure_pointer_t;
	typedef typename base_t::pure_ref_t pure_ref_t;

	/*!
	\brief Default constructor from any multithreaded smart pointer

	Locks pointed object.
	\param lp - multithreaded smart pointer
	*/
	explicit lsmart_ptr(const SP& lp)
		: base_t(lp), guard_(lp.mutex())
		, lobj_(*lp.mutex())
	{}

	/*!
	\brief Default constructor from any smart pointer and external mutex
	Use supplied external mutex to lock on. This ctor can be used with signlethreaded smart pointers.
	\param lp - smart pointer
	*/
	explicit lsmart_ptr(const SP& lp, bs_mutex& m)
		: base_t(lp), guard_(&m)
		, lobj_(m)
	{}

	//! copy ctor accuires another lock
	lsmart_ptr(const this_t& lp)
		: base_t(lp), guard_(lp.guard_)
		, lobj_(*lp.guard_)
	{}

	// destructor will automatically release the lock

	/*!
	\brief operator->

	Member-access operator returns modifiable object.
	\return pointer to non-constant object
	*/
	pure_pointer_t operator ->() const {
		return const_cast< pure_pointer_t >(this->p_);
	}

	/*!
	\brief conversion to non-constant simple pointer
	*/
	operator pure_pointer_t() const {
		return const_cast< pure_pointer_t >(this->p_);
	}

	/*!
	\brief Access to inner simple pointer.
	*/
	pure_pointer_t get() const {
		return const_cast< pure_pointer_t >(this->p_);
	}

	/*!
	\brief Dereferencing operator.
	\return Reference to modifiable object.
	*/
	pure_ref_t operator *() const {
		return const_cast< pure_ref_t >(*this->p_);
	}

	/*!
	\brief Assignment operator from reference to object

	Allows transparent assignment like 'ptr.lock() = a'
	\param r - reference to const object
	\return nonconst reference to pointed type.
	*/
	pure_ref_t operator =(ref_t r) const {
		*this->p_ = r;
		return *this->p_;
		//return const_cast< this_t& >(*this);
	}

	//there is no way to swap 2 lsmart_ptr

	//lock & unlock functions propagated to lock object
	/*!
	\brief Locks pointed object
	*/
	void lock() {
		lobj_.lock();
	}

	/*!
	\brief Unlocks pointed object
	*/
	void unlock() {
		lobj_.unlock();
		//fire unlock signal
		//bs_private::signal_unlock< pointed_t >::fire(this->p_);
	}

	/*!
	\brief Release lsmart_ptr.

	Also releases the lock.
	*/
	void release() {
		//release the lock
		lobj_.~lock_obj_t();
		//fire unlock signal
		//bs_private::signal_unlock< pointed_t >::fire(this->p_);
		//! release pointer
		base_t::release();
	}

	//dtor
	~lsmart_ptr() {
		//fire unlock signal
		//bs_private::signal_unlock< pointed_t >::fire(this->p_);
	}

private:
#if defined(_MSC_VER)
	friend class lsmart_ptr;
#else
	template< class R > friend class lsmart_ptr;
#endif

	bs_mutex* guard_;
	typedef bs_mutex::scoped_lock lock_obj_t;
	lock_obj_t lobj_;
};

/*!
\brief Function that performs the assignment using specified casting policy
Invoke this function if you want to assign smart pointers with non-default casting policy.
*/
template< class SP_l, class SP_r, bs_cast_policy cast_t >
SP_l& assign_sp(SP_l& lhs, const SP_r& rhs, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY()) {
	SP_l(rhs, cast).swap(lhs);
	return lhs;
}

template< class cast_t = BS_DEF_CAST_POLICY >
struct sp_assigner {
	template< class SP_l, class SP_r >
	SP_l& operator()(SP_l& lhs, const SP_r& rhs) {
		SP_l(rhs, cast_t()).swap(lhs);
		return lhs;
	}
};

// smart pointer template declaration
// tries to make compile-time checking for inheritance from blue_sky::bs_refcounter
// but such check works only for complete types
template
<
	class T,
	bool has_refcnt = conversion< T, bs_refcounter >::exists_uc
>
class smart_ptr;

/*!
class mt_ptr< T >
\ingroup smart_pointers
\brief most simple multithreaded pointer - not smart, doesn't contain reference counter
*/
template< class T >
class mt_ptr : public bs_private::smart_ptr_base< typename boost::add_const< T >::type >
{
public:
	typedef mt_ptr< T > this_t;
	typedef typename boost::add_const< T >::type pointed_t;
	typedef bs_private::smart_ptr_base< pointed_t > base_t;
	typedef pointed_t* pointer_t;
	typedef pointed_t& ref_t;
	typedef typename boost::remove_const< T >::type pure_pointed_t;
	typedef pure_pointed_t* pure_pointer_t;
	typedef pure_pointed_t& pure_ref_t;

	/*!
	\brief Constructor from simple pointer. By default pointed object doesn't deleted
	\param lp - pointer to const object
	\param mut - outer mutex used for access synchronization
	*/
	mt_ptr(pointer_t lp, bs_mutex& mut)
		: base_t(lp), mut_(&mut), d_(new bs_private::deleter_base())
	{}

	/*!
	\brief Constructor from simple pointer of any type and outer mutex
	*/
	template< class R, bs_cast_policy cast_t >
	mt_ptr(R* lp, bs_mutex& mut, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
		: base_t(lp, cast), mut_(&mut), d_(new bs_private::deleter_base())
	{}

	// explicitly define copy ctor
	// reason: folowing templated ctor DOES NOT prevent implicit copy ctor creation
	mt_ptr(const mt_ptr& lp)
		: base_t(lp), mut_(lp.mut_), d_(new bs_private::deleter_base())
	{}

	/*!
	\brief Templated constructor from mt_ptr to castable type
	Doesn't take ownership of object holded by lp
	*/
	template< class R >
	mt_ptr(const mt_ptr< R >& lp)
		: base_t(lp), mut_(lp.mut_), d_(new bs_private::deleter_base())
	{}

	template< class R, bs_cast_policy cast_t >
	mt_ptr(const mt_ptr< R >& lp, bs_castpol_val< cast_t > cast)
		: base_t(lp, cast), mut_(lp.mut_), d_(new bs_private::deleter_base())
	{}

	/*!
	\brief Templated constructor from smart_ptr to castable type
	*/
	template< class R >
	mt_ptr(const smart_ptr< R, true >& lp)
		: base_t(lp), mut_(*lp.mutex()),
		d_(new bs_private::deleter_adaptor< bs_obj_deleter< pointed_t > >())
	{
		if(lp) lp->add_ref();
	}

	~mt_ptr() {
		release();
	}

	/*!
	\brief Release inner pointer, making this empty (=NULL).
	*/
	void release() {
		if(this->p_) d_->dispose(this->p_);
		this->p_ = NULL;
	}

	/*!
	\brief Assignment from mt_ptr of castable type
	*/
	template< class R >
	this_t& operator=(const mt_ptr< R >& lp) {
		this_t(lp).swap(*this);
		return *this;
	}

	/*!
	\brief Assignment from smart_ptr to blue-sky object. Mutex embedded into pointed object used for synchronization
	*/
	template< class R >
	this_t& operator=(const smart_ptr< R, true >& lp) {
		this_t(lp).swap(*this);
		return *this;
	}

	/*!
	\brief Proxy function for accessing non-constant members of pointed object

	For quick non-const members access. See a little more detailed description in smart_ptr
	\return proxy locker object
	*/
//	const bs_locker< T > lock() const {
//		return bs_locker< T >(this->p_, *mut_);
//	}
	const lsmart_ptr< this_t > lock() const {
		return lsmart_ptr< this_t >(*this);
	}

	//! \brief mutex accessor
	bs_mutex* mutex() const { return mut_; }

	//! swap - swaps 2 mt_ptrs - never throws
	void swap(this_t& lp) {
		std::swap(this->p_, lp.p_);
		std::swap(mut_, lp.mut_);
		std::swap(d_, lp.d_);
	}

#ifdef BS_DISABLE_MT_LOCKS
	BS_SP_PURE_POINTER_ACCESS
#endif

private:
	friend class boost::serialization::access;
#if defined(_MSC_VER)
	friend class mt_ptr;
#else
	template< class R > friend class mt_ptr;
#endif

	//mutex
	mutable bs_mutex* mut_;
	//deleter
	bs_private::deleter_base* d_;
};

//----------------------------------------------------------------------------------------------------

/*!
class st_smart_ptr< T >
\ingroup smart_pointers
\brief single-threaded smart pointer for generic types with reference counter located in smart_ptr
*/
template< class T >
class st_smart_ptr : public bs_private::smart_ptr_base< T >
{
public:
	//typedefs
	typedef bs_private::smart_ptr_base< T > base_t;
	typedef st_smart_ptr< T > this_t;
	typedef typename base_t::pointed_t pointed_t;
	typedef typename base_t::pointer_t pointer_t;
	typedef typename base_t::ref_t ref_t;

	typedef bs_private::bs_refcounter_ptr< pointed_t, bs_refcounter_st > refcounter_ptr_t;
	//typedef boost_132::detail::shared_count refcounter_ptr_t;

	/*!
	\brief Constructor from simple pointer
	*/
	st_smart_ptr(pointer_t lp = NULL)
		: base_t(lp), count_(lp)
	{}

	/*!
	\brief Constructor from castable simple pointer

	Try to cast any pointer to specialized pointer.
	On reference counter reaches zero, object is destroyed via boost::checked_delete(R*).
	*/
	template< class R >
	st_smart_ptr(R* lp)
		: base_t(lp), count_(lp)
	{}

	template< class R, bs_cast_policy cast_t >
	st_smart_ptr(R* lp, bs_castpol_val< cast_t > cast)
		: base_t(lp, cast), count_(lp)
	{}

	/*!
	\brief Constructor with custom deleter.

	\param lp - any pointer
	\param d - custom deleter
	*/
	template< class R, class D, bs_cast_policy cast_t >
	st_smart_ptr(R* lp, D d, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
		: base_t(lp, cast), count_(lp, d)
	{}

	/*!
	\brief Constructor with custom deleter and allocator
	As above, but with allocator. A's copy constructor shall not throw.
	*/
	//template< class R, class D, class A, bs_cast_policy cast_t >
	//st_smart_ptr(R* lp, D d, A a, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
	//	: base_t(lp, cast), count_(lp, d, a)
	//{}

	/*!
	\brief Copy constructor - standard is fine
	*/
	//st_smart_ptr(const this_t& lp) throw()
	//	: base_t(lp), count_(lp.count_)
	//{
	//}

	/*!
	\brief Templated constructor from other generic smart_ptr of different type
	*/
	template< class R >
	st_smart_ptr(const st_smart_ptr< R >& lp)
		: base_t(lp), count_(lp.count_)
	{}

	template< class R, class cast_t >
	st_smart_ptr(const st_smart_ptr< R >& lp, cast_t cast)
		: base_t(lp, cast), count_(lp.count_)
	{}

	//! \brief standard destructor is fine with boost's shared_count
	//! \brief standard operator= is fine with boost's shared_count

	// this_t assignment
	//this_t& operator=(const this_t& lp) {
	//	this_t(lp).swap(*this);
	//	return *this;
	//}

	//! \brief template assignment operators

	//! \brief Assignment from castable simple pointer
	template< class R >
	this_t& operator=(R* lp) {
		//! assignment through swap
		this_t(lp).swap(*this);
		return *this;
	}

	/*!
	\brief Assignment operator from castable generic smart_ptr
	*/
	template< class R >
	this_t& operator=(const st_smart_ptr< R >& lp) {
		//! assignment through swap
		this_t(lp).swap(*this);
		return *this;
	}

	/*!
	\brief Release inner pointer, making this empty (=NULL).
	*/
	void release() {
		//! release through swap
		this_t().swap(*this);
	}

	//!	Get references count to pointed object.
	long refs() const {
		return count_.use_count();
		//return count_;
	}

	//! swap - swaps 2 smart_ptrs - never throws
	void swap(this_t& lp) {
		std::swap(this->p_, lp.p_);
		count_.swap(lp.count_);
	}

protected:
	friend class boost::serialization::access;
#if defined(_MSC_VER)
	friend class st_smart_ptr;
#else
	template< class R > friend class st_smart_ptr;
#endif

	// reference counter
	refcounter_ptr_t count_;
};

/*!
\class smart_ptr< T, false >
\ingroup smart_pointers
\brief multithreaded smart pointer for generic types
*/
template< class T >
class smart_ptr< T, false > : public st_smart_ptr< typename boost::add_const< T >::type >
{
public:
	typedef smart_ptr< T, false > this_t;
	typedef typename boost::add_const< T >::type pointed_t;
	typedef st_smart_ptr< pointed_t > base_t;
	typedef pointed_t* pointer_t;
	typedef pointed_t& ref_t;
	typedef typename boost::remove_const< T >::type pure_pointed_t;
	typedef pure_pointed_t* pure_pointer_t;
	typedef pure_pointed_t& pure_ref_t;
	// special typedef for boost::python
	// NOTE! breaks const+locks convention
	typedef T element_type;

	/*!
	\brief Constructor from simple pointer

	Default constructor which creates internal mutex.
	Deleter is boost::checked_delete(R*).
	\param lp - simple pointer
	*/
	smart_ptr(pointer_t lp = NULL)
		: base_t(lp), mut_(new bs_mutex), inner_(true)
	{}

	//mt_smart_ptr(pointer_t lp = NULL, bs_mutex& mut)
	//	: base_t(lp), mut_(&mut)
	//{}

	/*!
	\brief Constructor from castable simple pointer

	Construct from simple pointer and internal mutex as above.
	Deleter is boost::checked_delete(R*).
	*/
	template< class R >
	smart_ptr(R* lp)
		: base_t(lp), mut_(new bs_mutex), inner_(true)
	{}

	template< class R, bs_cast_policy cast_t >
	smart_ptr(R* lp, bs_castpol_val< cast_t > cast)
		: base_t(lp, cast), mut_(new bs_mutex), inner_(true)
	{}

	/*!
	\brief Constructor from castable simple pointer using external mutex

	Construct from simple pointer.
	Takes reference to external mutex.
	Deleter is boost::checked_delete(R*).
	*/
	template< class R, bs_cast_policy cast_t >
	smart_ptr(R* lp, bs_mutex& mut, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
		: base_t(lp, cast), mut_(&mut), inner_(false)
	{}

	/*!
	\brief Constructor from castable simple pointer using inner mutex and custom deleter
	As above but with custom deleter
	*/
	template< class R, class D, bs_cast_policy cast_t >
	smart_ptr(R* lp, D d, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
		: base_t(lp, d, cast), mut_(new bs_mutex), inner_(true)
	{}

	/*!
	\brief Constructor from castable simple pointer using external mutex and custom deleter
	As above but with external mutex
	*/
	template< class R, class D, bs_cast_policy cast_t >
	smart_ptr(R* lp, D d, bs_mutex& mut, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
		: base_t(lp, d, cast), mut_(&mut), inner_(false)
	{}

	/*!
	\brief Constructor from castable simple pointer using inner mutex, custom deleter and allocator

	As above, but with allocator. A's copy constructor shall not throw.
	Internal mutex.
	*/
	template< class R, class D, class A, bs_cast_policy cast_t >
	smart_ptr(R* lp, D d, A a, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
		: base_t(lp, d, a, cast), mut_(new bs_mutex), inner_(true)
	{}

	/*!
	\brief Constructor from castable simple pointer using external mutex, custom deleter and allocator

	As above, but with external mutex. A's copy constructor shall not throw.
	*/
	template< class R, class D, class A, bs_cast_policy cast_t >
	smart_ptr(R* lp, D d, A a, bs_mutex& mut, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
		: base_t(lp, d, a, cast), mut_(&mut), inner_(false)
	{}

	/*!
	\brief Templated constructor from simple multithreaded pointer of castable type
	*/
	template< class R >
	smart_ptr(const mt_ptr< R >& lp)
		: base_t(lp), mut_(lp.mutex()), inner_(false)
	{}

	template< class R, bs_cast_policy cast_t >
	smart_ptr(const mt_ptr< R >& lp, bs_castpol_val< cast_t > cast)
		: base_t(lp, cast), mut_(lp.mutex()), inner_(false)
	{}

	/*!
	\brief Templated constructor from generic single-threaded smart pointer of castable type
	Internal mutex.
	*/
	template< class R >
	smart_ptr(const st_smart_ptr< R >& lp)
		: base_t(lp), mut_(new bs_mutex), inner_(true)
	{}

	template< class R, bs_cast_policy cast_t >
	smart_ptr(const st_smart_ptr< R >& lp, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
		: base_t(lp, cast), mut_(new bs_mutex), inner_(true)
	{}

	/*!
	\brief Templated constructor from generic single-threaded smart pointer of castable type
	External mutex.
	*/
	template< class R, bs_cast_policy cast_t >
	smart_ptr(const st_smart_ptr< R >& lp, bs_mutex& mut, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
		: base_t(lp, cast), mut_(&mut), inner_(false)
	{}

	/*!
	\brief Constructor from castable smart pointer to any blue-sky type
	Uses custom deleter for blue-sky objects.
	*/
	template< class R, bs_cast_policy cast_t >
	smart_ptr(const smart_ptr< R, true >& lp, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
		: base_t(lp.get(), bs_obj_deleter< pointed_t >(), cast), inner_(false)
	{
		if(this->p_) {
			mut_ = &lp.mutex();
			lp->add_ref();
		}
		else //reset pointer
			this_t().swap(*this);
	}

	//standard copy constructor is fine
	//smart_ptr(const this_t& lp)
	//	: base_t(lp), mut_(lp.mut_)
	//{}

	/*!
	\brief Copy constructor from castable multithreaded smart_ptr.
	*/
	template< class R, bs_cast_policy cast_t >
	smart_ptr(const smart_ptr< R, false >& lp, bs_castpol_val< cast_t > cast = BS_DEF_CAST_POLICY())
		: base_t(lp, cast), mut_(lp.mut_), inner_(lp.inner_)
	{}

	//! destructor
	~smart_ptr() {
		//! delete mutex if necessary
		if(inner_ && this->refs() == 1)
			delete mut_;
	}

	//--------------other functions-------------------------------------------------------------------
	// standard operator= is fine with boost's shared_count

	/*!
	\brief Release inner pointer, making this empty (=NULL).
	*/
	void release() {
		//! release through swap
		this_t().swap(*this);
	}

	/*!
	\brief Assignment from simple pointer.
	This operation will keep mutex created on construction
	*/
	template< class R >
	this_t& operator=(const R* lp) {
		// assign through swap without touching inner falg
		this_t(lp, *mut_, bs_static_cast()).swap(*this, true);
		return *this;
	}

	/*!
	\brief Assignment from simple multithreaded pointer
	*/
	template< class R >
	this_t& operator=(const mt_ptr< R >& lp) {
		this_t(lp).swap(*this);
		return *this;
	}

	/*!
	\brief Assignment from castable smart_pointer of any type
	*/
	template< class R, bool pol >
	this_t& operator=(const smart_ptr< R, pol >& lp) {
		this_t(lp).swap(*this);
		return *this;
	}

	/*!
	\brief Assignment from castable single-threaded smart_ptr.
	This operation will keep mutex created on construction
	*/
	template< class R >
	this_t& operator=(const st_smart_ptr< R >& lp) {
		// assign through swap without touching inner falg
		this_t(lp, *mut_, bs_static_cast()).swap(*this, true);
		return *this;
	}

	/*!
	\brief Proxy function for accessing non-constant members of pointed object

	For quick non-const members access. See a little more detailed description in smart_ptr
	\return proxy locker object
	*/
//	const bs_locker< T > lock() const {
//		return bs_locker< T >(this->p_, *mut_);
//	}
	const lsmart_ptr< this_t > lock() const {
		return lsmart_ptr< this_t >(*this);
	}

	/*!
	\brief mutex accessor
	\return pointer to internal sp's mutex (bs_mutex)
	*/
	bs_mutex* mutex() const { return mut_; }

#ifdef BS_DISABLE_MT_LOCKS
	BS_SP_PURE_POINTER_ACCESS
#endif

private:
	friend class boost::serialization::access;
#if defined(_MSC_VER)
	friend class smart_ptr;
#else
	template< class R, bool pol > friend class smart_ptr;
#endif

	bs_mutex* mut_; //!< pointer to mutex object
	bool inner_; //!< is mutex internal (memory allocated in constructor)

	//! swap - swaps 2 mt_smart_ptrs - never throws
	void swap(this_t& lp, bool leave_inner_flag = false) {
		base_t::swap(lp);
		std::swap(mut_, lp.mut_);
		if(!leave_inner_flag)
			std::swap(inner_, lp.inner_);
	}

	//! deny conversion to smart_ptr< T >&
	operator st_smart_ptr< pure_pointed_t >&();
};

//----------------------------------------------------------------------------------------------------
/*!
class smart_ptr< T, true >
\ingroup smart_pointers
\brief Smart pointer for blue-sky objects (with inner reference count).
*/
template< class T >
class smart_ptr< T, true > : public bs_private::smart_ptr_base< typename boost::add_const<T>::type >
{
	//typedefs
public:
	//base smart_ptr type
	typedef bs_private::smart_ptr_base< typename boost::add_const< T >::type > base_t;
	//! type of *this object
	typedef smart_ptr< T, true > this_t;
	//! type of const T object
	typedef typename base_t::pointed_t pointed_t;
	//! type of pointer to const T object
	typedef typename base_t::pointer_t pointer_t;
	//! type of reference to const T object
	typedef typename base_t::ref_t ref_t;
	//! type of T object
	typedef typename boost::remove_const< T >::type pure_pointed_t;
	//! type of pointer to T object
	typedef pure_pointed_t* pure_pointer_t;
	//! type of reference to T object
	typedef pure_pointed_t& pure_ref_t;
	// special typedef for boost::python
	// NOTE! breaks const+locks convention
	typedef T element_type;

	/*!
	\brief Constructor.
	\param lp - simple pointer to const T
	*/
	/*explicit */smart_ptr(pointer_t lp = NULL) : base_t(lp) {
		//if(this->p_) this->p_->add_ref();
		bs_refcounter_add_ref (this->p_);
	};

	/*!
	\brief Copy constructor.
	*/
	smart_ptr(const this_t& lp) : base_t(lp) {
		//if(this->p_) this->p_->add_ref();
		bs_refcounter_add_ref (this->p_);
	}

	/*!
	\brief Templated Constructor from simple pointer.
	\param lp - any simple pointer. Default casting policy is used to determine if R can be casted to T
	*/
	template< class R >
	/*explicit */smart_ptr(R* lp) : base_t(lp)
	{
		//init(lp, Loki::Int2Type< cast_helper< R, BS_DEF_CAST_POLICY >::result >());
		//if(this->p_) this->p_->add_ref();
		bs_refcounter_add_ref (this->p_);
	}

	/*!
	\brief Templated constructor from simple pointer using custom casting policy.
	\param lp - any simple pointer. Supplied casting policy is used to determine if R can be casted to T
	*/
	template< class R, bs_cast_policy cast_t >
	/*explicit */smart_ptr(R* lp, bs_castpol_val< cast_t > cast) : base_t(lp, cast)
	{
		//init(lp, Loki::Int2Type< cast_helper< R, cast_t >::result >());
		//if(this->p_) this->p_->add_ref();
		bs_refcounter_add_ref (this->p_);
	}

	/*!
	\brief Templated constructor from smart_ptr of any type using default casting policy.
	*/
	template< class R >
	/*explicit */smart_ptr(const smart_ptr< R, true >& lp)
		: base_t(lp)
	{
		//init(lp, Loki::Int2Type< cast_helper< R, BS_DEF_CAST_POLICY >::result >());
		//if(this->p_) this->p_->add_ref();
		bs_refcounter_add_ref (this->p_);
	}

	/*!
	\brief Templated constructor from smart_ptr using custom casting policy
	*/
	template< class R, bs_cast_policy cast_t >
	/*explicit */smart_ptr(const smart_ptr< R, true >& lp, bs_castpol_val< cast_t > cast)
		: base_t(lp, cast)
	{
		//init(lp, Loki::Int2Type< cast_helper< R, cast_t >::result >());
		//if(this->p_) this->p_->add_ref();
		bs_refcounter_add_ref (this->p_);
	}

	/*!
	\brief Destructor.
	Dereferences pointed object.
	*/
	~smart_ptr() {
		if(this->p_) {
			//this->p_->del_ref();
			bs_refcounter_del_ref (static_cast <const bs_refcounter*> (this->p_));
		}
	}

	//assignment operators -----------------------------------------------------
	///*!
	//\brief Assignment from simple pointer of any type using custom casting policy
	//*/
	//template< class cast_t, class R >
	//this_t& assign(R* lp) throw() {
	//	this_t(lp, cast_t()).swap(*this);
	//	return *this;
	//}

	///*!
	//\brief Assignment from smart_ptr of any blue-sky using custom casting policy
	//*/
	//template< class cast_t, class R >
	//this_t& assign(const smart_ptr< R, true >& lp) throw() {
	//	this_t(lp, cast_t()).swap(*this);
	//	return *this;
	//}

	/*!
	\brief Assignment from another smart_ptr of the same type
	*/
	this_t& operator=(const this_t& lp) throw() {
		this_t(lp).swap(*this);
		return *this;
	}

	/*!
	\brief Assignment from smart_ptr of any blue-sky type using default casting policy
	*/
	template< class R >
	this_t& operator=(const smart_ptr< R, true >& lp) throw() {
		this_t(lp).swap(*this);
		return *this;
		//return sp_assigner< >()(*this, lp);
	}

	/*!
	\brief Assignment from simple pointer of any type using default casting policy
	*/
	template< class R >
	this_t& operator=(R* lp) throw() {
		this_t(lp).swap(*this);
		return *this;
		//return sp_assigner< >()(*this, lp);
	}

	//other functions------------------------------------------------------------
	/*!
	Get references count to pointed object.
	\return number of references
	*/
	long refs() const {
		if(this->p_) 
      return this->p_->refs();
		else return 0;
	}

	/*!
	\brief bool operator.
	Needed to allow constructions like 'if(smart_ptr<T>)'.
	*/
	//operator bool() const {
	//	return (this->p_ != NULL);
	//}

	/*!
	\brief function to access non-constant members of pointed object.
	Locks pointed object. Call this function before operator-> to access non-constant members.
	Intended for quick non-constant functions call with automatic object locking.
	\return proxy locker object
	*/
//	const bs_locker< T > lock() const {
//		return bs_locker< T >(this->p_);
//	}
	const lsmart_ptr< this_t > lock() const {
		return lsmart_ptr< this_t >(*this);
	}

	/*!
	\brief Makes this smart pointer empty (=NULL).
	*/
	void release() {
		//new version - through swap
		this_t().swap(*this);
	}

	/*!
	\brief Mutex accessor.
	\return pointer to object's mutex
	*/
	bs_mutex* mutex() const {
		if(this->p_)
			return &(this->p_->mutex());
		else return NULL;
	}

	//! swap - swaps 2 smart_ptrs - never throws
	void swap(this_t& lp) {
		std::swap(this->p_, lp.p_);
		//ref counters stay as is
	}

#ifdef BS_DISABLE_MT_LOCKS
	BS_SP_PURE_POINTER_ACCESS
#endif

private:
#if defined(_MSC_VER)
	friend class smart_ptr;
#else
	template< class R, bool > friend class smart_ptr;
#endif
};

namespace bs_private {

	//function that fires unlock signal
template< class T, bool can_fire >
struct signal_unlock {
	static void fire(T* p) {
		if(p) {
			//assume that unlock signal always has id = 1
			static_cast< const bs_imessaging* >(p)->fire_signal(1, sp_obj(NULL));
		}
	}
};

template< class T >
struct signal_unlock< T, false > {
	static void fire(T*) {}
};

}

//! -----------------------------------------comparison operators----------------------------------------------------

template< class T, class R >
bool inline operator ==(const bs_private::smart_ptr_base< T >& lp, const bs_private::smart_ptr_base< R >& rp) {
	return ((void*)lp.get() == (void*)rp.get());
}
template< class T, class R >
bool inline operator !=(const bs_private::smart_ptr_base< T >& lp, const bs_private::smart_ptr_base< R >& rp) {
	return !(lp == rp);
}
template< class T, class R >
bool inline operator <(const bs_private::smart_ptr_base< T >& lp, const bs_private::smart_ptr_base< R >& rp) {
	return ((void*)lp.get() < (void*)rp.get());
}

template< class T, class R >
bool inline operator ==(const bs_private::smart_ptr_base< T >& lp, const R* rp) {
	return ((void*)lp.get() == (void*)rp);
}
template< class T, class R >
bool inline operator !=(const bs_private::smart_ptr_base< T >& lp, const R* rp) {
	return !(lp == rp);
}
template< class T, class R >
bool inline operator <(const bs_private::smart_ptr_base< T >& lp, const R* rp) {
	return ((void*)lp.get() < (void*)rp);
}

template< class T, class R >
bool inline operator ==(const T* lp, const bs_private::smart_ptr_base< R >& rp) {
	return ((void*)lp == (void*)rp.get());
}
template< class T, class R >
bool inline operator !=(const T* lp, const bs_private::smart_ptr_base< R >& rp) {
	return !(lp == rp);
}
template< class T, class R >
bool inline operator <(const T* lp, const bs_private::smart_ptr_base< R >& rp) {
	return ((void*)lp < (void*)rp.get());
}

#if defined(BSPY_EXPORTING) || defined(BSPY_EXPORTING_PLUGIN)
// HACK! allow boost::python to obtain pure pointer via forced const_cast
// NOTE! this will break all locking rules in Python
// TODO: replace this with better solution
template <typename T>
inline T*
get_pointer(smart_ptr< T, true > const & p) {
	return const_cast< T* > (p.get());
}
#endif
} //end of namespace blue_sky

//#ifdef BSPY_EXPORTING
//// make pointee specializations to allow boost::python deduce type pointed to by smart_ptr
//// TODO: replace this with better solution
//namespace boost { namespace python {
//
//template < class T >
//struct pointee< blue_sky::smart_ptr< T, false > > {
//	typedef T type;
//};
//
//template < class T >
//struct pointee< blue_sky::smart_ptr< T, true > > {
//	typedef T type;
//};
//
//}}
//#endif

#endif /*_SMART_PTR_H*/
