/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/compat/messaging.h>
#include <bs/objbase.h>
#include <bs/kernel.h>
#include <bs/exception.h>

#include <boost/signals2.hpp>

using namespace std;
using namespace blue_sky;


namespace blue_sky {

BS_TYPE_IMPL(bs_signal, objbase, "bs_signal", "Compat BS signal", false, false)
BS_TYPE_ADD_CONSTRUCTOR(bs_signal, (int))

BS_TYPE_IMPL(bs_messaging, objbase, "bs_messaging", "Compat BS signals hub", true, true)
BS_TYPE_ADD_CONSTRUCTOR(bs_messaging, (const bs_messaging::sig_range_t&))

BS_REGISTER_TYPE(blue_sky::bs_signal)
BS_REGISTER_TYPE(blue_sky::bs_messaging)

/*-----------------------------------------------------------------
 * BS signal helpers
 *----------------------------------------------------------------*/
// only sync execution layer is supported
template< class slot_ptr >
class sync_layer {
public:
	static void fire_slot(
		const slot_ptr& slot, const sp_mobj& sender, int signal_code, const sp_obj& param
	) {
		if(slot)
			// directly execute slot
			slot->execute(sender, signal_code, param);
	}
};

// slot holder that uses execution layer to fire slot
template< class slot_ptr = sp_slot  >
class slot_holder : public sync_layer< slot_ptr > {
public:
	typedef const bs_imessaging* sender_ptr;
	typedef sync_layer< slot_ptr > base_t;
	using base_t::fire_slot;

	slot_holder(const slot_ptr& slot, const sender_ptr& sender = NULL)
		: slot_(slot), sender_(sender)
	{}

	bool operator==(const slot_holder& rhs) const {
		return (slot_ == rhs.slot_);
	}

	bool operator==(const slot_ptr& slot) const {
		return (slot_ == slot);
	}

	bool operator<(const slot_holder& rhs) const {
		return (slot_ < rhs.slot_);
	}

	void operator()(const sp_mobj& sender, int signal_code, sp_obj param) const {
		if((!sender_) || (sender.get() == sender_))
			fire_slot(slot_, sender, signal_code, param);
	}

protected:
	slot_ptr slot_;
	// if sender != NULL then only signals from this sender will be triggered
	// if we store sp_mobj then object will live forever, because every slot holds smart pointer to sender
	// thats why only pure pointer to object is stored
	// when sender is deleted, all slot_holders will be destroyed and there will be no dead references
	const bs_imessaging* sender_;
};

/*-----------------------------------------------------------------
 * BS signal implementation details
 *----------------------------------------------------------------*/
class bs_signal::signal_impl
{
public:
	//send signal command
	typedef boost::signals2::signal<
		void (const sp_mobj& sender, int signal_code, const sp_obj& param)
	> signal_engine;

	//default ctor
	signal_impl() : signal_code_(0) {}

	// ctor with sig code initialization
	signal_impl(int sig_code)
		: signal_code_(sig_code)
	{}

	void fire(const sp_mobj& sender, const sp_obj& param) {
		my_signal_(sender, signal_code_, param);
	}

	// if sender != NULL then slot will be activated only for given sender
	bool connect(const sp_slot& slot, const sp_mobj& sender = NULL) {
		if(!slot) return false;
		my_signal_.connect(slot_holder< >(slot, sender.get()));
		return true;
	}

	bool disconnect(const sp_slot& slot) {
		if(!slot) return false;
		my_signal_.disconnect(slot_holder< >(slot));
		return true;
	}

	ulong num_slots() const {
		return static_cast< ulong >(my_signal_.num_slots());
	}

	signal_engine my_signal_;
	int signal_code_;
};

//=============================== bs_signal implementation =============================================================
bs_signal::bs_signal(int signal_code)
	: pimpl_(new signal_impl(signal_code))
{}

void bs_signal::init(int signal_code) const {
	if(signal_code > 0)
		pimpl_->signal_code_ = signal_code;
	else
		throw bs_kexception("Wrong signal code given", "bs_signal::init");
}

void bs_signal::fire(const sp_mobj& sender, const sp_obj& param) const {
	//lock during recepients call because boost:signals doesn't support mt
	pimpl_->fire(sender, param);
}

bool bs_signal::connect(const sp_slot& slot, const sp_mobj& sender) const {
	return pimpl_->connect(slot, sender);
}

bool bs_signal::disconnect(const sp_slot& slot) const
{
	return pimpl_->disconnect(slot);
}

ulong bs_signal::num_slots() const {
	return pimpl_->num_slots();
}

int bs_signal::get_code() const {
	return pimpl_->signal_code_;
}

//============================== bs_messaging implementation ===========================================================
//empty ctor
bs_messaging::bs_messaging() {}

//ctor that adds all signals in a half-open range
bs_messaging::bs_messaging(const sig_range_t& sig_range) {
	add_signal(sig_range);
}

//copy ctor implementation
bs_messaging::bs_messaging(const bs_messaging& mes)
	: objbase(mes), signals_(mes.signals_)
{}

void bs_messaging::swap(bs_messaging& rhs) {
	std::swap(signals_, rhs.signals_);
}

//bs_messaging& bs_messaging::operator=(const bs_messaging& lhs) {
//	bs_messaging(lhs).swap(*this);
//	return *this;
//}

bool bs_messaging::fire_signal(int signal_code, const sp_obj& params) const {
	bs_signals_map::const_iterator sig = signals_.find(signal_code);
	if(sig == signals_.end()) return false;

	sig->second->fire(bs_shared_this< bs_messaging >(), params);
	return true;
}

bool bs_messaging::add_signal(int signal_code)
{
	if(signals_.find(signal_code) != signals_.end()) return false;

	sp_signal sig =  BS_KERNEL.create_object(bs_signal::bs_type(), signal_code);
	if(sig) {
		signals_[signal_code] = sig;
		return true;
	}
	return false;
}

ulong bs_messaging::add_signal(const sig_range_t& sr) {
	ulong cnt = 0;
	for(int i = sr.first; i < sr.second; ++i)
		cnt += static_cast< ulong >(add_signal(i));
	return cnt;
}

bool bs_messaging::remove_signal(int signal_code)
{
	bs_signals_map::const_iterator sig = signals_.find(signal_code);
	if(sig != signals_.end()) {
		signals_.erase(signal_code);
		return true;
	}
	return false;
}

std::vector< int > bs_messaging::get_signal_list() const
{
	std::vector< int > res;
	res.reserve(signals_.size());
	for(const auto& sig : signals_)
		res.emplace_back(sig.first);
	//for (bs_signals_map::const_iterator p = signals_.begin(); p != signals_.end(); ++p)
	//	res.push_back(p->first);
	return res;
}

bool bs_messaging::subscribe(int signal_code, const sp_slot& slot) const {
	if(!slot) return false;
	bs_signals_map::const_iterator sig = signals_.find(signal_code);
	if(sig != signals_.end()) {
		sig->second->connect(slot, bs_shared_this< bs_messaging >());
		return true;
	}
	return false;
}

bool bs_messaging::unsubscribe(int signal_code, const sp_slot& slot) const {
	if(!slot) return false;
	bs_signals_map::const_iterator sig = signals_.find(signal_code);
	if(sig != signals_.end()) {
		sig->second->disconnect(slot);
		return true;
	}
	return false;
}

ulong bs_messaging::num_slots(int signal_code) const {
	bs_signals_map::const_iterator sig = signals_.find(signal_code);
	if(sig != signals_.end())
		return sig->second->num_slots();
	else return 0;
}

}	//end of namespace blue_sky

