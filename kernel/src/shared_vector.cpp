/**
 * \file shared_vector.cpp
 * \brief
 * \author Sergey Miryanov
 * \date 03.11.2009
 * */
#include "shared_vector.h"
#include <iostream>

using namespace blue_sky;

template <typename T>
void
test_suite (T &t)
{
  t.push_back (0);
  t.push_back (1);
  t.push_back (2);

  t.insert (t.begin () + 1, 3, 3);
  t.insert (t.end (), 3, 4);
  t.insert (t.begin () + t.size () - 2, 4, 5);
  t.insert (t.begin (), 2, 6);
}

template <typename T>
void
test_suite_2 (T &t, T &t2)
{
  t2.push_back (7);
  t2.push_back (8);

  t.insert (t.begin () + 4, t2.begin (), t2.end ());
  t.insert (t.end (), t2.begin (), t2.end ());
  t.insert (t.begin (), t2.begin (), t2.end ());

  t2[0] = 11;
  t2[1] = 12;

  t.insert (t.end () - 2, t2.begin (), t2.end ());

  t2[0] = 13;
  t2[1] = 14;
  t.insert (t.end () - 1, t2.begin (), t2.end ());
}

template <typename T>
void
test_suite_3 (T &t)
{
  t.erase (t.begin () + 3);
  t.erase (t.begin () + 0);
  t.erase (t.end () - 1);

  t.erase (t.begin () + 0, t.begin () + 3);
  t.erase (t.begin () + 2, t.begin () + 3);
}

template <typename T>
void
test_suite_4 (T &t)
{
  t.resize (30, 0);
  t.resize (0, 0);
  t.resize (10, 9);
}

template <typename T>
void
test_suite_5 (T &t)
{
  T x (5, 15);
  t.clear ();
  t.insert (t.begin (), x.begin (), x.end ());
}

template <typename T>
void
test_suite_6 (T &t)
{
  T x (5, 15);
  T z (x.begin (), x.end ());

  t.clear ();
  t.insert (t.begin (), z.begin (), z.end ());
}

template <typename T>
void
test_suite_7 (T &t)
{
  T x;
  x.push_back (0);
  x.push_back (1);
  x.push_back (2);

  t.swap (x);
}

template <typename T>
void
test_suite_8 (T &t)
{
  t.assign (20, 4);
}

template <typename T>
void
test_suite_9 (T &t)
{
  T x;
  x.push_back (1);
  x.push_back (2);
  x.push_back (3);
  x.push_back (4);
  x.push_back (5);
  x.push_back (6);
  x.push_back (7);
  x.push_back (8);

  t.assign (x.begin (), x.end ());
}

template <typename T, typename X>
void
print (T &t, X &x)
{
  std::cout << "---" << std::endl;
  for (size_t i = 0; i < t.size (); ++i)
    {
      std::cout << "t[" << i << "] = " << t[i] << " - " << x[i] << std::endl;
      if (t[i] != x[i])
        {
          bs_throw_exception ("!!!");
        }
    }
  std::cout << "t.size () = " << t.size () << ", t.capacity = " << t.capacity () << std::endl;
  std::cout << "x.size () = " << x.size () << ", x.capacity = " << x.capacity () << std::endl;
}

void
test_shared_vector ()
{
  shared_vector <int> t, t2;
  std::vector <int> x, x2;

  test_suite (t);
  test_suite (x);
  print (t, x);

  test_suite_2 (t, t2);
  test_suite_2 (x, x2);
  print (t, x);

  test_suite_3 (t);
  test_suite_3 (x);
  print (t, x);

  test_suite_4 (t);
  test_suite_4 (x);
  print (t, x);

  test_suite_5 (t);
  test_suite_5 (x);
  print (t, x);

  test_suite_6 (t);
  test_suite_6 (x);
  print (t, x);

  test_suite_7 (t);
  test_suite_7 (x);
  print (t, x);

  test_suite_8 (t);
  test_suite_8 (x);
  print (t, x);

  test_suite_9 (t);
  test_suite_9 (x);
  print (t, x);
}
