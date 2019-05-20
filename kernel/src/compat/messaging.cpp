/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief 
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/compat/messaging.h>
#include <bs/kernel/types_factory.h>

#include <boost/signals2.hpp>

using namespace std;
using namespace blue_sky;


NAMESPACE_BEGIN(blue_sky)

BS_TYPE_IMPL(bs_signal, objbase, "bs_signal", "Compat BS signal", false, false)
BS_TYPE_ADD_CONSTRUCTOR(bs_signal, (int))

BS_TYPE_IMPL(bs_messaging, objbase, "bs_messaging", "Compat BS signals hub", true, true)
BS_TYPE_ADD_CONSTRUCTOR(bs_messaging, (const bs_messaging::sig_range_t&))

BS_REGISTER_TYPE("kernel", blue_sky::bs_signal)
BS_REGISTER_TYPE("kernel", blue_sky::bs_messaging)

// slot virtual dtor
bs_slot::~bs_slot() {}

/*-----------------------------------------------------------------
 * BS signal helpers
 *----------------------------------------------------------------*/
// only sync execution layer is supported
template< class slot_ptr >
class sync_layer {
public:
	static void fire_slot(
		const slot_ptr& slot, sp_cobj&& sender, int signal_code, std::any&& param
	) {
		if(slot) slot->execute(std::move(sender), signal_code, std::move(param));
	}
};

// slot holder that uses execution layer to fire slot
template< class slot_ptr = sp_slot  >
class slot_holder : public sync_layer< slot_ptr > {
public:
	using sender_ptr = std::weak_ptr<const objbase>;
	//typedef std::weak_ptr<const bs_imessaging> sender_ptr;
	typedef sync_layer< slot_ptr > base_t;
	using base_t::fire_slot;

	slot_holder(slot_ptr slot, const sp_cobj& sender = nullptr)
		: slot_(std::move(slot)), sender_(sender)
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

	void operator()(const sp_cobj& sender, int signal_code, const std::any& param) const {
		if(sender_.expired() || sender.get() == sender_.lock().get()) {
			// construct NEW temp copies of sender and param from source
			fire_slot(slot_, sp_cobj{sender}, signal_code, std::any{param});
		}
	}

protected:
	slot_ptr slot_;
	// if sender != nullptr then only signals from this sender will be triggered
	// if we store sp_obj then object will live forever, because every slot holds smart pointer to sender
	// thats why only pure pointer to object is stored
	// when sender is deleted, all slot_holders will be destroyed and there will be no dead references
	sender_ptr sender_;
};

/*-----------------------------------------------------------------
 * BS signal implementation details
 *----------------------------------------------------------------*/
class bs_signal::signal_impl
{
public:
	//send signal command
	typedef boost::signals2::signal<
		void (const sp_cobj& sender, int signal_code, const std::any& param)
	> signal_engine;

	//default ctor
	signal_impl() : signal_code_(0) {}

	// ctor with sig code initialization
	signal_impl(int sig_code)
		: signal_code_(sig_code)
	{}

	void fire(const sp_cobj& sender, std::any&& param) {
		my_signal_(sender, signal_code_, std::move(param));
	}

	// if sender != nullptr then slot will be activated only for given sender
	bool connect(const sp_slot& slot, const sp_cobj& sender = nullptr) {
		if(!slot) return false;
		my_signal_.connect(slot_holder<>(slot, sender));
		return true;
	}

	bool disconnect(const sp_slot& slot) {
		if(!slot) return false;
		my_signal_.disconnect(slot_holder<>(slot));
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

bs_signal::~bs_signal() = default;

void bs_signal::init(int signal_code) const {
	pimpl_->signal_code_ = signal_code;
}

void bs_signal::fire(const sp_cobj& sender, std::any param) const {
	pimpl_->fire(sender, std::move(param));
}

bool bs_signal::connect(const sp_slot& slot, const sp_cobj& sender) const {
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
// virtual imessaging dtor
bs_imessaging::~bs_imessaging() {}

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

bool bs_messaging::fire_signal(int signal_code, std::any param, const sp_cobj& sender) const {
	bs_signals_map::const_iterator sig = signals_.find(signal_code);
	if(sig == signals_.end()) return false;

	sig->second->fire(sender ? sender : shared_from_this(), std::move(param));
	return true;
}

bool bs_messaging::add_signal(int signal_code) {
	if(signals_.find(signal_code) != signals_.end()) return false;

	sp_signal sig =  kernel::tfactory::create_object(bs_signal::bs_type(), signal_code);
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
	return res;
}

bool bs_messaging::subscribe(int signal_code, const sp_slot& slot) const {
	if(!slot) return false;
	bs_signals_map::const_iterator sig = signals_.find(signal_code);
	if(sig != signals_.end()) {
		sig->second->connect(slot);
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

ulong bs_messaging::clear() {
	const ulong cnt = signals_.size();
	signals_.clear();
	return cnt;
}

NAMESPACE_END(blue_sky)
