#include "pch.h"

#include <boost/thread.hpp>

#include <networking/TreeResourceManager.h>
#include <networking/Path.h>
#include <networking/Session.h>
#include <networking/folder_listing.h>

#include <Request.h>
#include <Response.h>
#include <Http.h>

#include "bs_tree_helpers.h"
#include "LockService.h"

using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;
using namespace blue_sky::http_library;

class TreeResourceManager::Impl
{
	TreeResourceManager * fThis;
	LockService fLockService;
public:
	Impl (TreeResourceManager * trm)
	{
		fThis = trm;
	}

	//TreeContainer container;

	void process_objects_subtree(
		Session & context,
		Path & p,
		const Request & req,
		Response & resp)
	{
		static volatile int lock_count = 0;
		//TreeContainer::EntryPtr entry = container.Get(p);
		

		switch (req.method)
		{
		case Request::M_GET:
			{
				sp_link link = get_link(p);
				const bs_link * link_ptr = link.get();
				if (link_ptr && link_ptr->is_hard_link()) {			
					resp.Status_code = http::STATUS_200_OK;				
					resp.Body = link_ptr->data();
					std::ostrstream str;
					str << link_ptr->inode()->mtime() << std::ends;
					std::ostrstream str2;
					str2 << (void *) link_ptr->data().get() << std::ends;
					resp.ETag = str2.str();
				} else {
					resp.Status_code = http::STATUS_404_Not_Found;				
					resp.Reason_Phrase = "Object not found.";
				}
			}
			break;
		case Request::M_PUT:
			{
				delete_link(p);
				sp_link link = create_inode(p, req.Body);					
				//std::cout << "PUT " << fThis->path_prefix() << std::endl;
				if (link) {	
					resp.Status_code = http::STATUS_204_No_Content;					
				} else {
					resp.Status_code = http::STATUS_404_Not_Found;
				}
				
			}
			break;
		case Request::M_POST:
			resp.Status_code = http::STATUS_500_Internal_Server_Error;			
			resp.Reason_Phrase = "Not implemented.";
			break;
		case Request::M_DELETE:
			{						
				delete_link(p);
			}
			break;
		case Request::M_LOCK:
			{
				fLockService.lock(p.str());
				sp_link link = get_link(p);
				if (link && link->is_hard_link())
				{	
					
					//context.lock(p, link->data().mutex());
					//cout << "TRM::LOCKED " << p.str() << std::endl;
					resp.Status_code = http::STATUS_204_No_Content;
				}
				else
				{
					std::cerr << "Path not found: " << p.str() << std::endl;
					resp.Status_code = http::STATUS_404_Not_Found;
				}
			}
			break;
		case Request::M_UNLOCK:
			{			
				fLockService.unlock(p.str(), 0);
				//context.unlock(p);
				resp.Status_code = http::STATUS_204_No_Content;				
			}
			
			break;
		}
	}

	void process_listing_subtree(Session & context,
		Path & p,
		const Request & req,
		Response & resp)
	{
		switch (req.method)
		{
		case Request::M_GET:
			{
				
				sp_link link = get_link(p);
				//asdjflads
				const bs_link * link_ptr = link.get();
				if (link_ptr && link_ptr->is_node()) {		
					sp_node node = link_ptr->node();
					sp_folder_listing listing = BS_KERNEL.create_object(folder_listing::bs_type());
					
					std::vector<listing_entry_t> entries;
					for (bs_node::n_iterator it = node->begin(),
						end_it = node->end();
						it != end_it;
						++it)
					{
						listing_entry_t entry;
						entry.name = it->name();
						entry.isFolder = it->is_node();
						if (entry.isFolder == false)
						{
							entry.mtime = it->inode()->mtime();							
						} else 
						{
							entry.mtime = 0;
						}
						entries.push_back(entry);
					}
					{
						lsp_folder_listing locked(listing);
						locked->fill_from(entries);
					}
					resp.Status_code = http::STATUS_200_OK;				
					resp.Body = listing;
				} else {
					resp.Status_code = http::STATUS_400_Bad_Request;				
					resp.Reason_Phrase = p.str() + " is not folder.";
				}
			}
			break;
		default:
			resp.Status_code = http::STATUS_400_Bad_Request;
			resp.Reason_Phrase = "Not supported.";
		}
	}
};

TreeResourceManager::TreeResourceManager(
	ContextPtr context,
	const string & path_prefix)
: ResourceManager(context->name_service, path_prefix),
pimpl(new Impl(this))
{

};



void TreeResourceManager::process_request(
				Session & session,
				std::string const& path,
				const Request & req,
				Response & resp)
    {
        try
        {
			Path p(path);
			if (p.subfolderof("/objects/"))
			{
				p = p.subtract("/objects/");
				pimpl->process_objects_subtree(session, p, req, resp);
			} else if (p.equalto("/listing/") || p.subfolderof("/listing/"))
			{
				p = p.subtract("/listing/");
				pimpl->process_listing_subtree(session, p, req, resp);
			} else {
				resp.Status_code = http::STATUS_404_Not_Found;
				resp.Reason_Phrase = "Not found.";
			}
        }
        catch (std::exception & e)
        {			
			resp.Status_code = http::STATUS_500_Internal_Server_Error;
			resp.Reason_Phrase = string("Internal error.") + e.what();
        }
    }
