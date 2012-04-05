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

#include "bs_messaging.h"
#include "bs_object_base.h"
#include "bs_kernel.h"

#include "boost/signal.hpp"

#define SLOT_EXEC_LAYER sync_layer

using namespace std;
using namespace blue_sky;
using namespace Loki;

namespace blue_sky {

//---------------BlueSky Slot implementation--------------------------------------
//Andrey
//off
//struct bs_slot::slot_wrapper
//{
//	slot_wrapper(const sp_slot& slot) : slot_(slot)
//	{}
//
//	void operator()(sp_mobj sender, int signal_code, sp_obj param) const
//	{
//		kernel& k = give_kernel::Instance();
//		slot_.lock()->init(sender, signal_code, param);
//		k.add_task(slot_);
//	}
//
//	bool operator==(const slot_wrapper& rhs) const {
//		return (slot_ == rhs.slot_);
//	}
//
//	bool operator==(const sp_slot& slot) const {
//		return (slot_ == slot);
//	}
//
//	bool operator<(const slot_wrapper& rhs) const {
//		return (slot_ < rhs.slot_);
//	}
//
//	sp_slot slot_;
//};
//
//void bs_slot::init(const sp_mobj& sender, int signal_code, const sp_obj& param)
//{
//	sender_ = sender; signal_code_ = signal_code; param_ = param;
//}
//

void bs_slot::dispose() const {
	delete this;
}

//new implementation
class slot_holder {
public:
	typedef const bs_imessaging* sender_ptr;

	slot_holder(const sp_slot& slot, const sender_ptr& sender = NULL)
		: slot_(slot), sender_(sender)
	{}

	bool operator==(const slot_holder& rhs) const {
		return (slot_ == rhs.slot_);
	}

	bool operator==(const sp_slot& slot) const {
		return (slot_ == slot);
	}

	bool operator<(const slot_holder& rhs) const {
		return (slot_ < rhs.slot_);
	}

	void operator()(const sp_mobj& sender, int signal_code, sp_obj param) const {
		if((!sender_) || (sender == sender_))
			fire_slot(sender, signal_code, param);
	}

	// children should override this function
	virtual void fire_slot(const sp_mobj& sender, int signal_code, const sp_obj& param) const = 0;

	// virtual dtor
	virtual ~slot_holder() {};

protected:
	sp_slot slot_;
	// if sender != NULL then only signals from this sender will be triggered
	// if we store sp_mobj then object will live forever, because every slot holds smart pointer to sender
	// thats why only pure pointer to object is stored
	// when sender is deleted, all slot_holders will be destroyed and there will be no dead references
	const bs_imessaging* sender_;
	//sp_mobj sender_;
};

class async_layer : public slot_holder {
	class slot2com : public combase {
	public:
		slot2com(const sp_slot& slot, const sp_mobj& sender, int signal_code, const sp_obj& param)
			: slot_(slot), sender_(sender), sc_(signal_code), param_(param)
		{}

		sp_com execute() {
			slot_->execute(sender_, sc_, param_);
			return NULL;
		}

		void unexecute() {}
		bool can_unexecute() const { return false; }

		void dispose() const {
			//DEBUG
			//cout << "dispose() for slot_wrapper " << this << " called" << endl;
			delete this;
		}

	private:
		sp_slot slot_;
		sp_mobj sender_;
		int sc_;
		sp_obj param_;
	};

public:
	typedef slot_holder::sender_ptr sender_ptr;

	async_layer(const sp_slot& slot, const sender_ptr& sender = NULL)
		: slot_holder(slot, sender)
	{}

	void fire_slot(const sp_mobj& sender, int signal_code, const sp_obj& param) const {
		if(slot_)
			give_kernel::Instance().add_task(new slot2com(slot_, sender, signal_code, param));
	}
};

class sync_layer : public slot_holder {
public:
	typedef slot_holder::sender_ptr sender_ptr;

	sync_layer(const sp_slot& slot, const sender_ptr& sender = NULL)
		: slot_holder(slot, sender)
	{}

	void fire_slot(const sp_mobj& sender, int signal_code, const sp_obj& param) const {
		if(slot_)
			slot_->execute(sender, signal_code, param);
	}
};

//bs_signal definition
class bs_signal::signal_impl
{
public:
	//send signal command
	typedef boost::signal< void (const sp_mobj& sender, int signal_code, const sp_obj& param) > signal_engine;

	signal_engine my_signal_;
	//sp_mobj sender_;
	int signal_code_;
	sp_obj param_;

	//default ctor
	signal_impl() : signal_code_(0) {}

	// ctor with sig code initialization
	signal_impl(int sig_code)
		: signal_code_(sig_code)
	{}

//	signal_impl(const sp_mobj& sender, int sig_code)
//		: signal_code_(sig_code), sender_(sender),
//	{}

	//non-const function because boost::signals doesn't support mt
	void execute(const sp_mobj& sender) {
		if(sender)
			my_signal_(sender, signal_code_, param_);
	}

	void fire(const sp_mobj& sender, const sp_obj& param) {
		param_ = param;
		execute(sender);
	}

	// if sender != NULL then slot will be activated only for given sender
	bool connect(const sp_slot& slot, const sp_mobj& sender = NULL) {
		if(!slot) return false;
		//my_signal_.connect(bs_slot::slot_wrapper(slot));
		my_signal_.connect(SLOT_EXEC_LAYER(slot, sender));
		return true;
	}

	bool disconnect(const sp_slot& slot) {
		if(!slot) return false;
		my_signal_.disconnect(slot);
		return true;
	}

	ulong num_slots() const {
		return static_cast< ulong >(my_signal_.num_slots());
	}
};

//=============================== bs_signal implementation =============================================================
bs_signal::bs_signal(int signal_code)
	: pimpl_(new signal_impl(signal_code), mutex(), bs_static_cast())
{}

void bs_signal::init(int signal_code) const {
	lpimpl lp(pimpl_);
	if(signal_code > 0)
		lp->signal_code_ = signal_code;
	else
		throw bs_kernel_exception ("bs_signal::init", no_error, "Wrong signal code given");
}

//bool bs_signal::sender_binded() const {
//	return (pimpl_->sender_.get() != NULL);
//}

void bs_signal::fire(const sp_mobj& sender, const sp_obj& param) const {
	//lock during recepients call because boost:signals doesn't support mt
	//lpimpl lp(pimpl_);
	//lp->param_ = param;
	//lp->execute();
	pimpl_.lock()->fire(sender, param);
	//give_kernel::Instance().add_task(this);
}

bool bs_signal::connect(const sp_slot& slot, const sp_mobj& sender) const {
	return pimpl_.lock()->connect(slot, sender);
}

bool bs_signal::disconnect(const sp_slot& slot) const
{
	return pimpl_.lock()->disconnect(slot);
}

//sp_com bs_signal::execute()
//{
//	pimpl_->execute();
//	return NULL;
//}
//
//void bs_signal::unexecute()
//{}
//
//bool bs_signal::can_unexecute() const
//{
//	return false;
//}

ulong bs_signal::num_slots() const {
	return pimpl_->num_slots();
}

void bs_signal::dispose() const {
	delete this;
}


int bs_signal::get_code() const {
	return pimpl_->signal_code_;
}

//============================== bs_messaging implementation ===========================================================
//empty ctor
bs_messaging::bs_messaging() {}

//empty dtor
//bs_messaging::~bs_messaging() {}

//ctor that adds all signals in a half-open range
bs_messaging::bs_messaging(const sig_range_t& sig_range) {
	add_signal(sig_range);
}

//copy ctor implementation
bs_messaging::bs_messaging(const bs_messaging& mes)
	: bs_refcounter(mes), bs_imessaging(), signals_(mes.signals_)
{}

void bs_messaging::swap(bs_messaging& rhs) {
	std::swap(signals_, rhs.signals_);
}

//bs_messaging& bs_messaging::operator=(const bs_messaging& lhs) {
//	bs_messaging(lhs).swap(*this);
//	return *this;
//}

bool bs_messaging::fire_signal(int signal_code, const sp_obj& params) const
{
	bs_signals_map::const_iterator sig = signals_.find(signal_code);
	if(sig == signals_.end()) return false;

	//if(!sig->second->sender_binded())
	//	sig->second->init(this);
	sig->second->fire(this, params);
	//kernel &k = give_kernel::Instance();
	//sig->second.lock()->set_param(params);
	//k.add_task(sig->second);
	return true;
}

bool bs_messaging::add_signal(int signal_code)
{
	if(signals_.find(signal_code) != signals_.end()) return false;

	pair< sp_signal, bool > sig;
	sig =  BS_KERNEL.reg_signal(BS_GET_TI(*this), signal_code);
	signals_[signal_code] = sig.first;
	//new bs_signal(signal_code);
	return sig.second;
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
	if(sig != signals_.end())
	{
		signals_.erase(signal_code);
		BS_KERNEL.rem_signal(BS_GET_TI(*this), signal_code);
		return true;
	}
	return false;
}

std::vector< int > bs_messaging::get_signal_list() const
{
	std::vector< int > res;
	res.reserve(signals_.size());
	for (bs_signals_map::const_iterator p = signals_.begin(); p != signals_.end(); ++p)
		res.push_back(p->first);
	return res;
}

bool bs_messaging::subscribe(int signal_code, const sp_slot& slot) const {
	if(!slot) return false;
	bs_signals_map::const_iterator sig = signals_.find(signal_code);
	if(sig != signals_.end()) {
		sig->second->connect(slot, this);
		return true;
	}
	return false;
}

bool bs_messaging::unsubscribe(int signal_code, const smart_ptr< bs_slot, true >& slot) const
{
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

void bs_messaging::dispose() const {
	delete this;
}

}	//end of namespace blue_sky
