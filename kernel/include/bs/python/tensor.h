/// @file
/// @author uentity
/// @date 10.04.2019
/// @brief Eigen::Tensor Python bindings
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <pybind11/numpy.h>
#include "../meta/tensor.h"

// Eigen prior to 3.2.7 doesn't have proper move constructors--but worse, some classes get implicit
// move constructors that break things.  We could detect this an explicitly copy, but an extra copy
// of matrices seems highly undesirable.
static_assert(EIGEN_VERSION_AT_LEAST(3,2,7), "Eigen Tensor support in BS requires Eigen >= 3.2.7");

NAMESPACE_BEGIN(PYBIND11_NAMESPACE) NAMESPACE_BEGIN(detail)

#if EIGEN_VERSION_AT_LEAST(3,3,0)
using EigenIndex = Eigen::Index;
#else
using EigenIndex = EIGEN_DEFAULT_DENSE_INDEX_TYPE;
#endif

template<typename T> using TensorMeta = blue_sky::meta::tensor<T>;

// Helper struct for extracting information from tensor type
template <typename T> struct TensorProps {
	using Type = T;
	using Meta = TensorMeta<T>;
	using Traits = Eigen::internal::traits<Type>;
	static constexpr auto Ndims = Traits::NumDimensions;

	using Scalar = typename Traits::Scalar;
	using Index = typename Traits::Index;
	using DimsIndex = Eigen::array<EigenIndex, Ndims>;

	static constexpr bool
		row_major = Traits::Layout == Eigen::RowMajor,
		col_major = Traits::Layout == Eigen::ColMajor;

	using Array = array_t<Scalar, array::forcecast | (row_major ? array::c_style : array::f_style)>;

	static constexpr auto descriptor =
		_("numpy.ndarray[") + npy_format_descriptor<Scalar>::name +
		_<col_major>(", f_contiguous", "") +
		_<row_major>(", c_contiguous", "") +
		_<Meta::is_mutable>(", writeable", "") +
		_("]");
};

// Casts an Tensor type to numpy array.  If given a base, the numpy array references the src data,
// otherwise it'll make a copy.  writeable lets you turn off the writeable flag for the array.
template <typename TensorT> handle tensor_cast(TensorT const &src, handle base = handle(), bool writeable = true) {
	using props = TensorProps<TensorT>;
	using Array = typename props::Array;

	auto a = Array(src.dimensions(), src.data(), base);
	if (!writeable)
		array_proxy(a.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;
	return a.release();
}

// Takes an lvalue ref to some Tensor type and a (python) base object, creating a numpy array that
// reference the Tensor object's data with `base` as the python-registered base class (if omitted,
// the base will be set to None, and lifetime management is up to the caller).  The numpy array is
// non-writeable if the given type is const.
template <typename TensorT>
handle tensor_ref_array(TensorT &src, handle parent = none()) {
	// none here is to get past array's should-we-copy detection, which currently always
	// copies when there is no base.  Setting the base to None should be harmless.
	return tensor_cast(src, parent, !std::is_const<TensorT>::value);
}

// Takes a pointer to some dense, plain Tensor type, builds a capsule around it, then returns a numpy
// array that references the encapsulated data with a python-side reference to the capsule to tie
// its destruction to that of any dependent python objects.  Const-ness is determined by whether or
// not the Type of the pointer given is const.
template <typename TensorT>
handle tensor_encapsulate(TensorT *src) {
	capsule base(src, [](void *o) { delete static_cast<TensorT *>(o); });
	return tensor_ref_array(*src, base);
}

// Type caster for regular tensor, but not maps/refs/etc.
template<typename Type>
struct type_caster<Type, enable_if_t<TensorMeta<Type>::is_tensor>> {
	using Scalar = typename Type::Scalar;
	using props = TensorProps<Type>;
	using DimsIndex = typename props::DimsIndex;
	static constexpr auto Ndims = props::Ndims;

	bool load(handle src, bool convert) {
		// If we're in no-convert mode, only load if given an array of the correct type
		if (!convert && !isinstance<array_t<Scalar>>(src))
			return false;

		// Coerce into an array, but don't do type conversion yet; the copy below handles it.
		auto buf = array::ensure(src);

		if (!buf)
			return false;

		auto dims = buf.ndim();
		if (dims < 1) return false;

		// Allocate the new type, then build a numpy reference into it
		auto value_shape = DimsIndex{};
		std::copy_n(buf.shape(), buf.ndim(), value_shape.begin());
		value = Type(std::move(value_shape));
		auto ref = reinterpret_steal<array>(tensor_ref_array(value));
		if (dims == 1) ref = ref.squeeze();
		else if (ref.ndim() == 1) buf = buf.squeeze();

		int result = detail::npy_api::get().PyArray_CopyInto_(ref.ptr(), buf.ptr());

		if (result < 0) { // Copy failed!
			PyErr_Clear();
			return false;
		}

		return true;
	}

private:

	// Cast implementation
	template <typename CType>
	static handle cast_impl(CType *src, return_value_policy policy, handle parent) {
		switch (policy) {
			case return_value_policy::take_ownership:
			case return_value_policy::automatic:
				return tensor_encapsulate(src);
			case return_value_policy::move:
				return tensor_encapsulate(new CType(std::move(*src)));
			case return_value_policy::copy:
				return tensor_cast(*src);
			case return_value_policy::reference:
			case return_value_policy::automatic_reference:
				return tensor_ref_array(*src);
			case return_value_policy::reference_internal:
				return tensor_ref_array(*src, parent);
			default:
				throw cast_error("unhandled return_value_policy: should not happen!");
		};
	}

public:

	// Normal returned non-reference, non-const value:
	static handle cast(Type &&src, return_value_policy /* policy */, handle parent) {
		return cast_impl(&src, return_value_policy::move, parent);
	}
	// If you return a non-reference const, we mark the numpy array readonly:
	static handle cast(const Type &&src, return_value_policy /* policy */, handle parent) {
		return cast_impl(&src, return_value_policy::move, parent);
	}
	// lvalue reference return; default (automatic) becomes copy
	static handle cast(Type &src, return_value_policy policy, handle parent) {
		if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
			policy = return_value_policy::copy;
		return cast_impl(&src, policy, parent);
	}
	// const lvalue reference return; default (automatic) becomes copy
	static handle cast(const Type &src, return_value_policy policy, handle parent) {
		if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
			policy = return_value_policy::copy;
		return cast(&src, policy, parent);
	}
	// non-const pointer return
	static handle cast(Type *src, return_value_policy policy, handle parent) {
		return cast_impl(src, policy, parent);
	}
	// const pointer return
	static handle cast(const Type *src, return_value_policy policy, handle parent) {
		return cast_impl(src, policy, parent);
	}

	static constexpr auto name = props::descriptor;

	operator Type*() { return &value; }
	operator Type&() { return value; }
	operator Type&&() && { return std::move(value); }
	template <typename T> using cast_op_type = movable_cast_op_type<T>;

private:
	Type value;
};

// Base class for casting reference/map/block/etc. objects back to python.
template <typename MapType> struct tensor_map_caster {
private:
	using props = TensorProps<MapType>;
	using meta =  typename props::Meta;

public:

	// Directly referencing a ref/map's data is a bit dangerous (whatever the map/ref points to has
	// to stay around), but we'll allow it under the assumption that you know what you're doing (and
	// have an appropriate keep_alive in place).  We return a numpy array pointing directly at the
	// ref's data (The numpy array ends up read-only if the ref was to a const matrix type.) Note
	// that this means you need to ensure you don't destroy the object in some other way (e.g. with
	// an appropriate keep_alive, or with a reference to a statically allocated matrix).
	static handle cast(const MapType &src, return_value_policy policy, handle parent) {
		switch (policy) {
			case return_value_policy::copy:
				return tensor_cast(src);
			case return_value_policy::reference_internal:
				return tensor_cast(src, parent, meta::is_mutable_handle);
			case return_value_policy::reference:
			case return_value_policy::automatic:
			case return_value_policy::automatic_reference:
				return tensor_cast(src, none(), meta::is_mutable_handle);
			default:
				// move, take_ownership don't make any sense for a ref/map:
				pybind11_fail("Invalid return_value_policy for Tensor Map/Ref/Block type");
		}
	}

	static constexpr auto name = props::descriptor;

	// Explicitly delete these: support python -> C++ conversion on these (i.e. these can be return
	// types but not bound arguments).  We still provide them (with an explicitly delete) so that
	// you end up here if you try anyway.
	bool load(handle, bool) = delete;
	operator MapType() = delete;
	template <typename> using cast_op_type = MapType;
};

// We can return any map-like object (but can only load Refs, specialized next):
template <typename Type> struct type_caster<Type, enable_if_t<TensorMeta<Type>::is_map>>
	: tensor_map_caster<Type> {};

// Loader for Ref<...> arguments.  See the documentation for info on how to make this work without
// copying (it requires some extra effort in many cases).
template <typename PlainObjectType>
struct type_caster<Eigen::TensorRef<PlainObjectType>> : public tensor_map_caster<Eigen::TensorRef<PlainObjectType>> {
private:
	using Type = Eigen::TensorRef<PlainObjectType>;
	using MapType = Eigen::TensorMap<PlainObjectType>;
	using props = TensorProps<Type>;
	using meta = typename props::Meta;
	using Scalar = typename props::Scalar;
	using DimsIndex = typename props::DimsIndex;
	using Array = typename props::Array;

	static constexpr auto Ndims = props::Ndims;
	static constexpr bool need_writeable = meta::is_mutable_ref;

	// Delay construction (these have no default constructor)
	std::unique_ptr<MapType> map;
	std::unique_ptr<Type> ref;
	// Our array.  When possible, this is just a numpy array pointing to the source data, but
	// sometimes we can't avoid copying (e.g. input is not a numpy array at all, has an incompatible
	// layout, or is an array of a type that needs to be converted).  Using a numpy temporary
	// (rather than an Eigen temporary) saves an extra copy when we need both type conversion and
	// storage order conversion.  (Note that we refuse to use this temporary copy when loading an
	// argument for a Ref<M> with M non-const, i.e. a read-write reference).
	Array copy_or_ref;
public:
	bool load(handle src, bool convert) {
		// First check whether what we have is already an array of the right type.  If not, we can't
		// avoid a copy (because the copy is also going to do type conversion).
		bool need_copy = !isinstance<Array>(src);

		if (!need_copy) {
			// We don't need a converting copy, but we also need to check whether the strides are
			// compatible with the Ref's stride requirements
			Array aref = reinterpret_borrow<Array>(src);

			if (aref && (!need_writeable || aref.writeable())) {
				copy_or_ref = std::move(aref);
			}
			else {
				need_copy = true;
			}
		}

		if (need_copy) {
			// We need to copy: If we need a mutable reference, or we're not supposed to convert
			// (either because we're in the no-convert overload pass, or because we're explicitly
			// instructed not to copy (via `py::arg().noconvert()`) we have to fail loading.
			if (!convert || need_writeable) return false;

			Array copy = Array::ensure(src);
			if (!copy) return false;
			copy_or_ref = std::move(copy);
			loader_life_support::add_patient(copy_or_ref);
		}

		ref.reset();
		DimsIndex map_shape;
		std::copy_n(copy_or_ref.shape(), copy_or_ref.ndim(), map_shape.begin());
		map.reset(new MapType(data(copy_or_ref), map_shape));
		ref.reset(new Type(*map));

		return true;
	}

	operator Type*() { return ref.get(); }
	operator Type&() { return *ref; }
	template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;

private:
	template <typename T = Type, enable_if_t<TensorMeta<T>::is_mutable, int> = 0>
	Scalar *data(Array &a) { return a.mutable_data(); }

	template <typename T = Type, enable_if_t<!TensorMeta<T>::is_mutable, int> = 0>
	const Scalar *data(Array &a) { return a.data(); }
};

NAMESPACE_END(detail) NAMESPACE_END(PYBIND11_NAMESPACE)
