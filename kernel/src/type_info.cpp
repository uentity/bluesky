/// @author uentity
/// @date 28.04.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/type_descriptor.h>
#include <bs/defaults.h>
#include <bs/objbase.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(detail)

NAMESPACE_END(detail)

NAMESPACE_BEGIN()
/*-----------------------------------------------------------------------------
 *  Nil type tag
 *-----------------------------------------------------------------------------*/
struct nil_ {
	static auto bs_type() -> const type_descriptor& {
		static const auto nil_td = type_descriptor(
			defaults::nil_type_name, nullptr, nullptr, nullptr, "Nil type"
		);
		return nil_td;
	}
};

NAMESPACE_END()

const std::type_index& nil_type_info() {
	static const auto nil_ti = BS_GET_TI(nil_);
	return nil_ti;
}

bool is_nil(const std::type_index& t) {
	return t == std::type_index(typeid(nil_));
}

/*-----------------------------------------------------------------------------
 *  type_descriptor impl
 *-----------------------------------------------------------------------------*/
// constructor from string type name for temporary tasks (searching etc)
type_descriptor::type_descriptor(std::string_view type_name) :
	parent_td_fun_(&nil), assign_fun_(detail::noop_assigner),
	copy_fun_(nullptr), name(type_name)
{}

// standard constructor
type_descriptor::type_descriptor(
	std::string type_name, BS_GET_TD_FUN parent_td_fn, BS_TYPE_ASSIGN_FUN assign_fn,
	BS_TYPE_COPY_FUN cp_fn, std::string description
) :
	parent_td_fun_(parent_td_fn ? parent_td_fn : &nil),
	assign_fun_(assign_fn ? assign_fn : detail::noop_assigner), copy_fun_(cp_fn),
	name(std::move(type_name)), description(std::move(description))
{}

// obtain Nil type_descriptor instance
auto type_descriptor::nil() -> const type_descriptor& {
	return nil_::bs_type();
}

auto type_descriptor::is_nil() const -> bool {
	return this == &nil();
}

auto type_descriptor::is_copyable() const -> bool {
	return (copy_fun_ != nullptr);
}

auto type_descriptor::parent_td() const -> const type_descriptor& {
	return parent_td_fun_();
}

auto type_descriptor::operator <(const type_descriptor& td) const -> bool {
	return name < td.name;
}

auto type_descriptor::clone(bs_type_copy_param src) const -> shared_ptr_cast {
	return copy_fun_ ? (*copy_fun_)(src) : nullptr; 
}

auto type_descriptor::assign(sp_obj target, sp_obj source, prop::propdict params) const -> error {
	return assign_fun_(std::move(target), std::move(source), std::move(params));
}

auto type_descriptor::isinstance(const sp_cobj& obj) const -> bool {
	return obj->bs_resolve_type() == *this;
}

auto isinstance(const sp_cobj& obj, std::string_view obj_type_id) -> bool {
	return obj->bs_resolve_type() == obj_type_id;
}

auto upcastable_eq::operator()(const type_descriptor& td1, const type_descriptor& td2) const -> bool {
	if(td1 == td2) return true;

	const type_descriptor* cur_td = &td2.parent_td();
	while(!cur_td->is_nil()) {
		if(td1 == *cur_td)
			return true;
		cur_td = &cur_td->parent_td();
	}
	return false;
}

NAMESPACE_END(blue_sky)
