#pragma once

#include <fstream>
#include <vector>
#include <string>

#include "definitions.hpp"
/*
 * class for data output
 * called every iteration. an internal counter
 * will check if the current state should
 * be stored.
 *
 * outputs are buffered (_store) until a write is
 * requested, either through trigger (_dump_every) 
 * or by calling flush()
*/
template<class State>
struct gnuplot_obs
{
  std::string _filename;
  size_t _store_every, _dump_every;
  size_t _count;
  
  std::vector<std::tuple<size_t, Real, Real, State>> _store; 
  std::ofstream _ofs;

  gnuplot_obs( std::string filename, size_t store_every = 10, size_t dump_every = 0 )
  : _filename(std::move(filename)), _store_every( store_every ) , _dump_every(dump_every), _count( 0 ), _store(_dump_every) 
  {
    _ofs.open(_filename);
    _ofs << "# output of oregonator\n";
    _ofs << "step\ttime\ttstep\tX\tY\tZ\n";
    _ofs.close();
  }

  void operator()( const State &x , Real t, Real dt )
  {
    if( ( _count % _store_every ) == 0 )
    {
      _store.emplace_back(_count, t, dt, x);
    }

    if( _dump_every > 0 && _count % _dump_every == 0 )
    {
      flush();
    }
    ++_count;
  }

  void flush()
  {
    _ofs.open(_filename, std::ofstream::app);
      std::for_each(std::begin(_store), std::end(_store), [&](auto&& e)
      {
        _ofs << std::get<0>(e) 
            << "\t" << std::get<1>(e) 
            << "\t" << std::get<2>(e) 
            << "\t" << std::get<3>(e)[0] 
            << "\t" << std::get<3>(e)[1] 
            << "\t" << std::get<3>(e)[2] 
            << "\n";
      });
      _ofs.close();
      _store.clear();
  }
};


