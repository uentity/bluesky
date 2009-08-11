#include "pch.h"
#include "bs_tree_helpers.h"



using namespace std;
using namespace boost;
using namespace blue_sky;
using namespace blue_sky::networking;

sp_node blue_sky::networking::get_node(const Path & path)
{
	kernel & k = BS_KERNEL;
	sp_link link_sp = k.bs_root();
	const bs_link * link = link_sp.get();
	sp_link current_link = link_sp;
	for (Path::iterator it = path.begin(),
		end_it = path.end();
		it != end_it;
	++it)
	{
		if (current_link->is_node())
		{
			sp_node node = current_link->node();
			const bs_node *  node_ptr = node.get();
			bs_node::n_iterator nit = node_ptr->find(*it);

			if (nit == node_ptr->end())
				//Или все таки исключение кинуть?
				return sp_node();
			current_link = nit.operator->();
		}
		else
		{
			return sp_node();
		}
	}
	if (current_link->is_node())
		return current_link->node();
	else
		return sp_node();
}

sp_link blue_sky::networking::get_link(const Path & path)
{
	kernel & k = BS_KERNEL;
	const sp_link link_sp = k.bs_root();
	const bs_link * link = link_sp.get();
	sp_link current_link = link_sp;
	for (Path::iterator it = path.begin(),
		end_it = path.end();
		it != end_it;
		++it)
		{
			if (current_link->is_node())
			{
				sp_node node = current_link->node();
				const bs_node *  node_ptr = node.get();
				bs_node::n_iterator nit = node_ptr->find(*it);

				if (nit == node_ptr->end())
					//Или все таки исключение кинуть?
					return sp_link();
				current_link = nit.operator->();
			}
			else
			{
				return sp_link();
			}
		}
	return current_link;
}

bool blue_sky::networking::exists(const Path & path)
{
	kernel & k = BS_KERNEL;
	sp_link link_sp = k.bs_root();
	const bs_link * link = link_sp.get();
	sp_link current_link = link_sp;
	for (Path::iterator it = path.begin(),
		end_it = path.end();
		it != end_it;
	++it)
	{
		if (current_link->is_node())
		{
			sp_node node = current_link->node();
			const bs_node *  node_ptr = node.get();
			bs_node::n_iterator nit = node_ptr->find(*it);

			if (nit == node_ptr->end())
				return false;
			current_link = nit.operator->();
		}
		else
		{
			return false;
		}
	}
	return true;
}

void blue_sky::networking::delete_link(const Path & path)
{
	kernel & k = BS_KERNEL;
	sp_link link_sp = k.bs_root();
	const bs_link * link = link_sp.get();
	sp_node parent_node;
	sp_link current_link = link_sp;
	for (Path::iterator it = path.begin(),
		end_it = path.end();
		it != end_it;
	++it)
	{
		if (current_link->is_node())
		{
			sp_node node = current_link->node();
			const bs_node *  node_ptr = node.get();
			bs_node::n_iterator nit = node_ptr->find(*it);

			if (nit == node_ptr->end())
				return;
			parent_node = node;
			current_link = nit.operator->();
		}
		else
		{
			return;
		}
	}
	if (!parent_node)
		//Не дадим удалить "/"
		//Или все-таки исключение?
		return;
	
	const bs_node * parent_node_ptr = parent_node.get();
	parent_node_ptr->erase(current_link);
}

sp_link create_inode(sp_obj obj, sp_node attach_to,  Path::const_iterator & begin, Path::const_iterator & end)
{
	Path fullPath(begin, end);
	Path dirPath = fullPath.up();
	Path fName = fullPath.subtract(dirPath);
	
	sp_node parent_node = attach_to;

	for (Path::const_iterator it = dirPath.begin(),
		end_it = dirPath.end();
		it != end_it;
		++it)
		{
			sp_node node = bs_node::create_node();
			parent_node->insert(node, *it);
			parent_node = node;
		}
	
	sp_link link(bs_link::create(obj, *fName.begin()));
	parent_node->insert(link);	
	return link;
}

sp_link blue_sky::networking::create_inode(const Path & path, sp_obj obj)
{
	kernel & k = BS_KERNEL;
	sp_link link_sp = k.bs_root();
	const bs_link * link = link_sp.get();
	sp_link current_link = link_sp;
	Path p = path.up();
	for (Path::iterator it = path.begin(),
		end_it = path.end();
		it != end_it;
	++it)
	{
		if (current_link->is_node())
		{			
			sp_node node = current_link->node();
			const bs_node *  node_ptr = node.get();
			bs_node::n_iterator nit = node_ptr->find(*it);

			if (nit == node_ptr->end())
			{				
				return ::create_inode(obj, node, it, end_it);				
			}
			current_link = nit.operator->();
		}
		else
		{
			//Ну тут уж точно надо исключение...
			return sp_link();
		}
	}
	return sp_link();
}