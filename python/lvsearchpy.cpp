#include <boost/python.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/to_python_converter.hpp>
#include <boost/python/overloads.hpp>
#include "container_conversions.h"
#include "lv_search.h"

#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

using namespace boost::python;
namespace bp = boost::python;

template<class T>
struct VecToList
{
  static PyObject* convert(const std::vector<T>& vec){
    boost::python::list* l = new boost::python::list();
    for(size_t i =0; i < vec.size(); i++)
      (*l).append(vec[i]);

    return l->ptr();
  }
};

// lvsearchpy module definitions

static std::vector<double> wrap_llh(LVSearch* lv,std::vector<double> arg){
  if(arg.size() != 3)
    throw std::runtime_error("Number of arguments should be 3. You sent me " + std::to_string(arg.size()));
  std::array<double,3> argv;
  std::copy_n(arg.begin(),3,argv.begin());
  auto fr = lv->llh(argv);
  std::vector<double> result = fr.params;
  result.push_back(fr.likelihood);
  return result;
}

static double wrap_llhFull(LVSearch* lv,std::vector<double> arg){
  if(arg.size() != 9)
    throw std::runtime_error("Number of arguments should be 3. You sent me " + std::to_string(arg.size()));
  std::array<double,9> argv;
  std::copy_n(arg.begin(),9,argv.begin());
  double llh = lv->llhFull(argv);
  return llh;
}

BOOST_PYTHON_MODULE(lvsearchpy)
{
  // import numpy array definitions
  import_array();
  import_ufunc();

  class_<LVSearch, boost::noncopyable, std::shared_ptr<LVSearch> >("LVSearch", init<std::string,std::string,std::string,std::string,std::string,std::string>())
    .def("llh",wrap_llh)
    .def("llhFull",wrap_llhFull)
    .def("SetVerbose",&LVSearch::SetVerbose)
    ;

  // python container to vector<double> convertion
  using namespace scitbx::boost_python::container_conversions;
  from_python_sequence< std::vector<double>, variable_capacity_policy >();
  to_python_converter< std::vector<double, class std::allocator<double> >, VecToList<double> > ();
}
