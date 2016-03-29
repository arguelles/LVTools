#ifndef AUTODIFF_H_INCLUDED
#define AUTODIFF_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/math/constants/constants.hpp>

//namespace autodiff{
	
	const int Dynamic=-1;
	
	//fwd decl
	template <int nVars, typename T>
	class FD;
	
	namespace detail{
		
		//Provide a uniform mechanism for assessing the dimensionality of a FD's gradient
		//For fixed-size FDs, just return the constant size
		template <template<int,class> class F, int D, class T>
		struct dimensionExtractor{
			static unsigned int nVars(const F<D,T>&){
				return(F<D,T>::N);
			}
		};
		
		//For dynamically-sized FDs, actually query the object's current dimensionality
		template <template<int,class> class F, class T>
		struct dimensionExtractor<F,Dynamic,T>{
			static unsigned int nVars(const F<Dynamic,T>& f){
				return(f.nVars);
			}
		};
		
	}
	
	///\brief A simple class for forward-mode automatic differentiation
	///
	///If the number of independent variables derivatives (or better, the number
	///of independent variables for which the derivatives of the computed values
	///are interesting) is known at compile time, this should be supplied as the
	///first template parameter for this class, allowing dynamic memory allocation
	///to be avoided. This can make computation more efficient by up to a factor
	///of 20-30. Otherwise, the special value `Dynamic` may be used, which casues
	///storage for gradient vectors to be allocated on the heap.
	///
	///Each of the independent
	///variables for which partial derivatives are to be evaluated must be assigned
	///a unique position within the gradient vector, using assignIndex() (on the
	///independent variable). Finally, the partial derivatives of a dependent
	///variable with respect to the independent variables are read out after the
	///calculation using derivative() on the dependent variable, where the index
	///argument to derivative() is the index assigned to the independent variable
	///with respect to which the partial derivative is desired.
	///
	///Example:
	/// Suppose we want to compute f(x,y) = 2*x*y + 4*x + 6*y, df/dx, and df/dy
	/// at x=7.22, y=9.4, in single precision.
	/// First, we create the independent variables, and assign each an index in
	/// the gradient vector:
	/// > autodiff::FD<2,float> x(7.22);
	/// > autodiff::FD<2,float> y(9.4);
	/// > x.assignIndex(0);
	/// > y.assignIndex(1);
	/// Or, more concisely:
	/// > autodiff::FD<2,float> x(7.22,0);
	/// > autodiff::FD<2,float> x(9.4,1);
	/// Then we calculate the result (and implicitly calculate the derivatives):
	/// > autodiff::FD<2,float> f = 2.0f*x*y + 4.0f*x + 6.0f*z;
	/// We can then get the value and the derivatives:
	/// > std::cout << "f = " << f.value() << std::endl;
	/// > std::cout << "df/dx = " << f.derivative(0) << std::endl;
	/// > std::cout << "df/dy = " << f.derivative(1) << std::endl;
	/// The same can be done even if the total number of independent variables is
	/// not known ahead of time (perhaps because this is part of some larger
	/// calculation for which the number of included variables can change):
	/// > autodiff::FD<autodiff::Dynamic,float> x(7.22,0);
	/// > autodiff::FD<autodiff::Dynamic,float> x(9.4,1);
	/// > autodiff::FD<autodiff::Dynamic,float> f = 2.0f*x*y + 4.0f*x + 6.0f*z;
	/// > std::cout << "f = " << f.value() << std::endl;
	/// > std::cout << "df/dx = " << f.derivative(0) << std::endl;
	/// > std::cout << "df/dy = " << f.derivative(1) << std::endl;
	/// Note that it is still necessary to assign unique indices to the independent
	/// variables, however, this works perfectly well with indices chosen at runtime.
	template <int nVars, typename T=double>
	class FD{
	private:
		T v; //value
		T g[nVars]; //gradient
		
		/*static constexpr struct noInit_t{} noInit{};
		
		explicit FD<nVars,T>(const noInit_t&){}
		
		FD<nVars,T>(T t, const noInit_t&):
		v(t){}*/
		
	public:
		typedef T BaseType;
		enum {N=nVars};
		
		FD<nVars,T>(){
			std::fill(&g[0],&g[0]+nVars,0);
		}
		
		FD<nVars,T>(T t):
		v(t){
			std::fill(&g[0],&g[0]+nVars,0);
		}
		
		FD<nVars,T>(T t, unsigned int index):
		v(t){
			assert(index<nVars);
			std::fill(&g[0],&g[0]+nVars,0);
			g[index]=T(1);
		}
		
		FD<nVars,T>(const FD<nVars,T>& f):
		v(f.v){
			std::copy(&f.g[0],&f.g[0]+nVars,&g[0]);
		}
		
		FD<nVars,T>& operator=(const FD<nVars,T>& f){
			if(&f!=this){
				v=f.v;
				std::copy(&f.g[0],&f.g[0]+nVars,&g[0]);
			}
			return(*this);
		}
		
		FD<nVars,T>& operator=(const T& t){
			v=t;
			std::fill(&g[0],&g[0]+nVars,0);
			return(*this);
		}
		
		void assignIndex(unsigned int index){
			assert(index<nVars);
			std::fill(&g[0],&g[0]+nVars,0);
			g[index]=T(1);
		}
		
		explicit
		operator T() const{
			return(v);
		}
		
		const T& value() const{
			return(v);
		}
		
		const T& derivative(unsigned int index) const{
			assert(index<nVars);
			return(g[index]);
		}
		
		void copyGradient(T grad[nVars]) const{
			std::copy(&g[0],&g[0]+nVars,&grad[0]);
		}
		
		void setDerivative(T d, unsigned int index){
			g[index]=d;
		}
		
		//unary +
		FD<nVars,T> operator+() const{
			return(*this);
		}
		
		//unary -
		FD<nVars,T> operator-() const{
			FD<nVars,T> r(*this);
			r.v=-r.v;
			std::transform(&r.g[0],&r.g[0]+nVars,&r.g[0],std::negate<T>());
			return(r);
		}
		
		//addition
		template <typename U, typename=typename std::enable_if<std::is_constructible<T,U>::value>::type>
		FD<nVars,T>& operator+=(const U& u){
			v+=T(u);
			return(*this);
		}
		
		FD<nVars,T>& operator+=(const FD<nVars,T>& f){
			v+=f.v;
			std::transform(&g[0],&g[0]+nVars,&f.g[0],&g[0],std::plus<T>());
			return(*this);
		}
		
		template <typename U, typename=typename std::enable_if<std::is_constructible<T,U>::value>::type>
		FD<nVars,T> operator+(const U& u) const{
			return(FD<nVars,T>(*this)+=u);
		}
		
		FD<nVars,T> operator+(const FD<nVars,T>& f) const{
			//return(FD<nVars,T>(*this)+=f);
			//for(unsigned int i=0; i<nVars; i++)
			//	result.g[i] = g[i] + f.g[i];
			FD<nVars,T> result(v+f.v/*,noInit*/);
			std::transform(&g[0],&g[0]+nVars,&f.g[0],&result.g[0],std::plus<T>());
			return(result);
		}
		
		//subtraction
		template <typename U, typename=typename std::enable_if<std::is_constructible<T,U>::value>::type>
		FD<nVars,T>& operator-=(const U& u){
			v-=T(u);
			return(*this);
		}
		
		FD<nVars,T>& operator-=(const FD<nVars,T>& f){
			v-=f.v;
			std::transform(&g[0],&g[0]+nVars,&f.g[0],&g[0],std::minus<T>());
			return(*this);
		}
		
		template <typename U, typename=typename std::enable_if<std::is_constructible<T,U>::value>::type>
		FD<nVars,T> operator-(const U& u) const{
			return(FD<nVars,T>(*this)-=u);
		}
		
		FD<nVars,T> operator-(const FD<nVars,T>& f) const{
			//return(FD<nVars,T>(*this)-=f);
			//for(unsigned int i=0; i<nVars; i++)
			//	result.g[i] = g[i] - f.g[i];
			FD<nVars,T> result(v-f.v/*,noInit*/);
			std::transform(&g[0],&g[0]+nVars,&f.g[0],&result.g[0],std::minus<T>());
			return(result);
		}
		
		//multiplication
		template <typename U, typename=typename std::enable_if<std::is_constructible<T,U>::value>::type>
		FD<nVars,T>& operator*=(const U& u){
			T t(u);
			v*=t;
			std::transform(&g[0],&g[0]+nVars,&g[0],std::bind2nd(std::multiplies<T>(),t));
			return(*this);
		}
		
		FD<nVars,T>& operator*=(const FD<nVars,T>& f){
			for(unsigned int i=0; i<nVars; i++)
				g[i] = g[i]*f.v + f.g[i]*v;
			v*=f.v;
			return(*this);
		}
		
		template<typename U, typename=typename std::enable_if<std::is_constructible<T,U>::value>::type>
		FD<nVars,T> operator*(const U& u) const{
			return(FD<nVars,T>(*this)*=u);
		}
		
		FD<nVars,T> operator*(const FD<nVars,T>& f) const{
			//return(FD<nVars,T>(*this)*=f);
			FD<nVars,T> result(v*f.v/*,noInit*/);
			for(unsigned int i=0; i<nVars; i++)
				result.g[i] = g[i]*f.v + f.g[i]*v;
			return(result);
		}
		
		//division
		template<typename U, typename=typename std::enable_if<std::is_constructible<T,U>::value>::type>
		FD<nVars,T>& operator/=(const U& u){
			T t(u);
			v/=t;
			std::transform(&g[0],&g[0]+nVars,&g[0],std::bind2nd(std::divides<T>(),t));
			return(*this);
		}
		
		FD<nVars,T>& operator/=(const FD<nVars,T>& f){
			v/=f.v;
			for(unsigned int i=0; i<nVars; i++)
				g[i] = (g[i] - v*f.g[i])/f.v;
			return(*this);
		}
		
		template<typename U, typename=typename std::enable_if<std::is_constructible<T,U>::value>::type>
		FD<nVars,T> operator/(const U& u) const{
			return(FD<nVars,T>(*this)/=u);
		}
		
		FD<nVars,T> operator/(const FD<nVars,T>& f) const{
			//return(FD<nVars,T>(*this)/=f);
			FD<nVars,T> result(v/f.v/*,noInit*/);
			for(unsigned int i=0; i<nVars; i++)
				result.g[i] = (g[i] - result.v*f.g[i])/f.v;
			return(result);
		}
		
		template <template<int,class> class F, int D, class T_>
		friend struct detail::dimensionExtractor;
		template <int nVars_, typename T_, typename U>
		friend FD<nVars_,T_> operator/(const U& u, const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> cos(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> sin(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> tan(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> acos(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> asin(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> atan(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> cosh(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> sinh(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> tanh(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> exp(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> log(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> log10(const FD<nVars_,T_>& f);
		template <typename FD_t, typename U>
		friend FD_t pow(const FD_t& b, const U& e, typename boost::enable_if< boost::is_arithmetic< U >, int >::type);
		template <typename FD_t, typename U>
		friend FD_t pow(const U& b, const FD_t& e, typename boost::enable_if< boost::is_arithmetic< U >, int >::type);
		template <typename FD_t, typename U>
		friend FD_t pow(const U& b, const FD_t& e, typename boost::enable_if< boost::is_arithmetic< U >, int >::type);
		template <typename FD_t>
		friend FD_t pow(const FD_t& b, const FD_t& e);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> sqrt(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> floor(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> ceil(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> abs(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> fabs(const FD<nVars_,T_>& f);
		
	private:
		//this is a no-op, provided only for uniformity of interface with FD<Dynamic>
		void changeGradientSize(unsigned int newVars){}
	};
	
	template <int nVars, typename T, typename U>
	FD<nVars,T> operator+(const U& u, const FD<nVars,T>& f){
		return(FD<nVars,T>(f)+=u);
	}
	
	template <int nVars, typename T, typename U>
	FD<nVars,T> operator-(const U& u, const FD<nVars,T>& f){
		return(-(FD<nVars,T>(f)-=u));
	}
	
	template <int nVars, typename T, typename U>
	FD<nVars,T> operator*(const U& u, const FD<nVars,T>& f){
		return(FD<nVars,T>(f)*=u);
	}
	
	template <int nVars, typename T, typename U>
	FD<nVars,T> operator/(const U& u, const FD<nVars,T>& f){
		//return(detail::div_impl<nVars,T,U>::div(u,f));
		FD<nVars,T> result(f);
		T t(u);
		result.v=t/result.v;
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),-result.v/f.v));
		return(result);
	}
	
	//set up wrapper functions to sort out overload resolution issues
/*#define ad_make_wrapper(func) \
template<typename T> \
typename boost::enable_if <boost::is_fundamental<T>,T>::type \
func ## _wrapper(const T& t){ return(::func(t)); } \
template<typename T> \
typename boost::disable_if <boost::is_fundamental<T>,T>::type \
func ## _wrapper(const T& t){ return(func(t)); }
#define ad_make_wrapper2(func) \
template<typename T, typename U> \
typename boost::enable_if <boost::mpl::and_<boost::is_fundamental<T>,boost::is_fundamental<U> >,T>::type \
func ## _wrapper(const T& t, const U& u){ return(std::func(t,u)); } \
template<typename T, typename U> \
typename boost::disable_if <boost::mpl::and_<boost::is_fundamental<T>,boost::is_fundamental<U> >,T>::type \
func ## _wrapper(const T& t, const U& u){ return(func(t,u)); }
	
	ad_make_wrapper(cos)
	ad_make_wrapper(sin)
	ad_make_wrapper(tan)
	ad_make_wrapper(acos)
	ad_make_wrapper(asin)
	ad_make_wrapper(atan)
	ad_make_wrapper(cosh)
	ad_make_wrapper(sinh)
	ad_make_wrapper(tanh)
	ad_make_wrapper(exp)
	ad_make_wrapper(log)
	ad_make_wrapper(log10)
	ad_make_wrapper2(pow)
	ad_make_wrapper(sqrt)
	ad_make_wrapper(ceil)
	ad_make_wrapper(floor)
	
#undef ad_make_wrapper
#undef ad_make_wrapper2*/
	
	//pull normal definitions of these functions into the namespace so that they won't be totally shadowed by the locally defined templates
	using std::cos;
	using std::sin;
	using std::tan;
	using std::acos;
	using std::asin;
	using std::atan;
	using std::cosh;
	using std::sinh;
	using std::tanh;
	using std::exp;
	using std::pow;
	using std::log;
	using std::log10;
	using std::sqrt;
	using std::ceil;
	using std::floor;
	
	template <int nVars, typename T>
	FD<nVars,T> cos(const FD<nVars,T>& f){
		FD<nVars,T> result(cos(f.v)/*,FD<nVars,T>::noInit*/);
		const T m(-sin(f.v));
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&f.g[0],&f.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),m));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> sin(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=sin(f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		const T m(cos(f.v));
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),m));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> tan(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=tan(f.v);
		T c(cos(f.v));
		T m=1/(c*c);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),m));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> acos(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=acos(f.v);
		T m=-T(1)/sqrt(T(1)-f.v*f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),m));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> asin(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=asin(f.v);
		T m=T(1)/sqrt(T(1)-f.v*f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),m));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> atan(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=atan(f.v);
		T m=T(1)/(T(1)+f.v*f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),m));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> cosh(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=cosh(f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),sinh(f.v)));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> sinh(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=sinh(f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),cosh(f.v)));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> tanh(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=tanh(f.v);
		T c(cosh(f.v));
		T m=T(1)/(c*c);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),m));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> exp(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=exp(f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),result.v));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> log(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=log(f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::divides<T>(),f.v));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> log10(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=log10(f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::divides<T>(),log(T(10))*f.v));
		return(result);
	}
	
	/*template <int nVars, typename T, typename U>
	 FD<nVars,T> pow(const FD<nVars,T>& b, const U& e){
	 FD<nVars,T> result(b);
	 T te(e);
	 result.v=pow(b.v,te);
	 std::transform(&result.g[0],&result.g[0]+nVars,&result.g[0],
	 std::bind2nd(std::multiplies<T>(),e*pow(b.v,te-T(1))));
	 return(result);
	 }
	 
	 template <int nVars, typename T, typename U>
	 FD<nVars,T> pow(const U& b, const FD<nVars,T>& e){
	 FD<nVars,T> result(e);
	 T tb(b);
	 result.v=pow(tb,e.v);
	 std::transform(&result.g[0],&result.g[0]+nVars,&result.g[0],
	 std::bind2nd(std::multiplies<T>(),result.v*log(tb)));
	 return(result);
	 }
	 
	 template <int nVars, typename T>
	 FD<nVars,T> pow(const FD<nVars,T>& b, const FD<nVars,T>& e){
	 FD<nVars,T> result(pow(b.v,e.v));
	 T c1(e.v*pow(b.v,e.v-T(1)));
	 T c2(result.v*log(b.v));
	 for(unsigned int i=0; i<nVars; i++)
	 result.g[i] = c1*b.g[i] + c2*e.g[i];
	 return(result);
	 }*/
	template <typename FD_t, typename U>
	FD_t pow(const FD_t& b, const U& e, typename boost::enable_if< boost::is_arithmetic< U >, int >::type = 0){
		typedef typename FD_t::BaseType T;
		FD_t result(b);
		T te(e);
		result.v=pow(b.v,te);
		const unsigned int n=detail::dimensionExtractor<FD,FD_t::N,typename FD_t::BaseType>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),pow(b.v,te-T(1))*e));
		return(result);
	}
	
	template <typename FD_t, typename U>
	FD_t pow(const U& b, const FD_t& e, typename boost::enable_if< boost::is_arithmetic< U >, int >::type = 0){
		typedef typename FD_t::BaseType T;
		FD_t result(e);
		T tb(b);
		result.v=pow(tb,e.v);
		const unsigned int n=detail::dimensionExtractor<FD,FD_t::N,typename FD_t::BaseType>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::multiplies<T>(),result.v*log(tb)));
		return(result);
	}
	
	template <typename FD_t>
	FD_t pow(const FD_t& b, const FD_t& e){
		typedef typename FD_t::BaseType T;
		FD_t result(pow(b.v,e.v));
		//the result needs as much space in its gradient vector as the larger of b or e
		result.changeGradientSize(std::max(detail::dimensionExtractor<FD,FD_t::N,typename FD_t::BaseType>::nVars(b),
		                                   detail::dimensionExtractor<FD,FD_t::N,typename FD_t::BaseType>::nVars(e)));
		T c1(e.v*pow(b.v,e.v-T(1)));
		T c2(result.v*log(b.v));
		const unsigned int n=detail::dimensionExtractor<FD,FD_t::N,typename FD_t::BaseType>::nVars(result);
		for(unsigned int i=0; i<n; i++)
			result.g[i] = c1*b.g[i] + c2*e.g[i];
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> sqrt(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=sqrt(f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
					   std::bind2nd(std::divides<T>(),T(2)*result.v));
		return(result);
	}

	template <typename FD_t, typename U>
	FD_t atan2(const FD_t& x, const U& y, typename boost::enable_if< boost::is_arithmetic< U >, int >::type = 0){
		if(x==0){
			if(y>0)
				return(FD_t(boost::math::constants::half_pi<typename FD_t::BaseType>()));
			if(y<0)
				return(FD_t(-boost::math::constants::half_pi<typename FD_t::BaseType>()));
			if(std::numeric_limits<typename FD_t::BaseType>::has_quiet_NaN)
				return(FD_t(std::numeric_limits<typename FD_t::BaseType>::quiet_NaN()));
			if(std::numeric_limits<typename FD_t::BaseType>::has_signaling_NaN)
				return(FD_t(std::numeric_limits<typename FD_t::BaseType>::signaling_NaN()));
			throw std::domain_error("x==0 in calls to atan2()");
		}
		FD_t result=atan(x/y);
		if(x<0){
			if(y>=0)
				result+=boost::math::constants::pi<typename FD_t::BaseType>();
			else
				result-=boost::math::constants::pi<typename FD_t::BaseType>();
		}
		return(result);
	}

	template <typename FD_t, typename U>
	FD_t atan2(const U& x, const FD_t& y, typename boost::enable_if< boost::is_arithmetic< U >, int >::type = 0){
		if(x==0){
			if(y>0)
				return(FD_t(boost::math::constants::half_pi<typename FD_t::BaseType>()));
			if(y<0)
				return(FD_t(-boost::math::constants::half_pi<typename FD_t::BaseType>()));
			if(std::numeric_limits<typename FD_t::BaseType>::has_quiet_NaN)
				return(FD_t(std::numeric_limits<typename FD_t::BaseType>::quiet_NaN()));
			if(std::numeric_limits<typename FD_t::BaseType>::has_signaling_NaN)
				return(FD_t(std::numeric_limits<typename FD_t::BaseType>::signaling_NaN()));
			throw std::domain_error("x==0 in calls to atan2()");
		}
		FD_t result=atan(x/y);
		if(x<0){
			if(y>=0)
				result+=boost::math::constants::pi<typename FD_t::BaseType>();
			else
				result-=boost::math::constants::pi<typename FD_t::BaseType>();
		}
		return(result);
	}

	template <int nVars, typename T>
	FD<nVars,T> atan2(const FD<nVars,T>& x, const FD<nVars,T>& y){
		if(x==0){
			if(y>0)
				return(FD<nVars,T>(boost::math::constants::half_pi<T>()));
			if(y<0)
				return(FD<nVars,T>(-boost::math::constants::half_pi<T>()));
			if(std::numeric_limits<T>::has_quiet_NaN)
				return(FD<nVars,T>(std::numeric_limits<T>::quiet_NaN()));
			if(std::numeric_limits<T>::has_signaling_NaN)
				return(FD<nVars,T>(std::numeric_limits<T>::signaling_NaN()));
			throw std::domain_error("x==0 in calls to atan2()");
		}
		FD<nVars,T> result=atan(x/y);
		if(x<0){
			if(y>=0)
				result+=boost::math::constants::pi<T>();
			else
				result-=boost::math::constants::pi<T>();
		}
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> ceil(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=ceil(f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		//TODO: is it true that (ignoring discontinuities) the derivative of ceil is insensitive to variations in all variables?
		std::fill(&result.g[0],&result.g[0]+n,T(0));
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> abs(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		bool flip=f.v<0;
		if(flip){
			result.v=-result.v;
			const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
			std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
						   std::bind2nd(std::multiplies<T>(),T(-1)));
		}
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> fabs(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		bool flip=f.v<0;
		if(flip){
			result.v=-result.v;
			const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
			std::transform(&result.g[0],&result.g[0]+n,&result.g[0],
						   std::bind2nd(std::multiplies<T>(),T(-1)));
		}
		return(result);
	}
	
	template <int nVars, typename T>
	FD<nVars,T> floor(const FD<nVars,T>& f){
		FD<nVars,T> result(f);
		result.v=floor(f.v);
		const unsigned int n=detail::dimensionExtractor<FD,nVars,T>::nVars(result);
		//TODO: is it true that (ignoring discontinuities) the derivative of floor is insensitive to variations in all variables?
		std::fill(&result.g[0],&result.g[0]+n,T(0));
		return(result);
	}
	
	//comparison operators:
	template <int nVars, typename T>
	bool operator!(const FD<nVars,T>& f){
		return(!f.value());
	}
	
	//equality
	template <int nVars, typename T, typename U>
	bool operator==(const FD<nVars,T>& f, const U& u){
		return(f.value()==u);
	}
	
	template <int nVars, typename T, typename U>
	bool operator==(const U& u, const FD<nVars,T>& f){
		return(u==f.value());
	}
	
	template <int nVars, typename T>
	bool operator==(const FD<nVars,T>& f1, const FD<nVars,T>& f2){
		return(f1.value()==f2.value());
	}
	
	//inequality
	template <int nVars, typename T, typename U>
	bool operator!=(const FD<nVars,T>& f, const U& u){
		return(f.value()!=u);
	}
	
	template <int nVars, typename T, typename U>
	bool operator!=(const U& u, const FD<nVars,T>& f){
		return(u!=f.value());
	}
	
	template <int nVars, typename T>
	bool operator!=(const FD<nVars,T>& f1, const FD<nVars,T>& f2){
		return(f1.value()!=f2.value());
	}
	
	//greater-than
	template <int nVars, typename T, typename U>
	bool operator>(const FD<nVars,T>& f, const U& u){
		return(f.value()>u);
	}
	
	template <int nVars, typename T, typename U>
	bool operator>(const U& u, const FD<nVars,T>& f){
		return(u>f.value());
	}
	
	template <int nVars, typename T>
	bool operator>(const FD<nVars,T>& f1, const FD<nVars,T>& f2){
		return(f1.value()>f2.value());
	}
	
	//greater-than-or-equal
	template <int nVars, typename T, typename U>
	bool operator>=(const FD<nVars,T>& f, const U& u){
		return(f.value()>=u);
	}
	
	template <int nVars, typename T, typename U>
	bool operator>=(const U& u, const FD<nVars,T>& f){
		return(u>=f.value());
	}
	
	template <int nVars, typename T>
	bool operator>=(const FD<nVars,T>& f1, const FD<nVars,T>& f2){
		return(f1.value()>=f2.value());
	}
	
	//less-than
	template <int nVars, typename T, typename U>
	bool operator<(const FD<nVars,T>& f, const U& u){
		return(f.value()<u);
	}
	
	template <int nVars, typename T, typename U>
	bool operator<(const U& u, const FD<nVars,T>& f){
		return(u<f.value());
	}
	
	template <int nVars, typename T>
	bool operator<(const FD<nVars,T>& f1, const FD<nVars,T>& f2){
		return(f1.value()<f2.value());
	}
	
	//less-than-or-equal
	template <int nVars, typename T, typename U>
	bool operator<=(const FD<nVars,T>& f, const U& u){
		return(f.value()<=u);
	}
	
	template <int nVars, typename T, typename U>
	bool operator<=(const U& u, const FD<nVars,T>& f){
		return(u<=f.value());
	}
	
	template <int nVars, typename T>
	bool operator<=(const FD<nVars,T>& f1, const FD<nVars,T>& f2){
		return(f1.value()<=f2.value());
	}
	
	//stream operators
	template <int nVars, typename T>
	std::ostream& operator<<(std::ostream& os, const FD<nVars,T>& f){
		os << f.value() << '[';
		for(unsigned int i=0; i<nVars; i++){
			if(i)
				os << ',';
			os << f.derivative(i);
		}
		os << ']';
		return(os);
		//return(os << f.value());
	}
	
	template <int nVars, typename T>
	std::istream& operator>>(std::istream& is, FD<nVars,T>& f){
		T v;
		is >> v;
		f = v;
		char dummy;
		if(!is.good())
			return(is);
		if(is.peek()!='[')
			return(is);
		is >> dummy;
		for(unsigned int i=0; i<nVars; i++){
			if(i)
				is >> dummy;
			is >> v;
			f+=v*FD<nVars,T>(0,i);
		}
		is >> dummy;
		return(is);
	}
	
	//TODO: could this be made allocator aware?
	template <typename T>
	class FD<Dynamic,T>{
	private:
		T v; //value
		T* g; //gradient
		unsigned int nVars;
		
		inline void changeGradientSize(unsigned int newVars){
			if(newVars<=nVars)
				return;
			T* newg = new T[newVars];
			//copy over any data we already had
			std::copy(&g[0],&g[0]+nVars,&newg[0]);
			//zero fill the new portion of the gradient
			std::fill(&newg[0]+nVars,&newg[0]+newVars,T(0));
			delete[] g;
			g=newg;
			nVars=newVars;
		}
		
		static constexpr struct noInit_t{} noInit{};
		
		FD<Dynamic,T>(T t, noInit_t):
		v(t),g(NULL),nVars(0){}
		
	public:
		typedef T BaseType;
		enum{N=-1};
		
		FD<Dynamic,T>():g(NULL),nVars(0){}
		
		FD<Dynamic,T>(T t):
		v(t),g(NULL),nVars(0){}
		
		FD<Dynamic,T>(T t, unsigned int index):
		v(t),g(NULL),nVars(0){
			changeGradientSize(index+1);
			g[index]=T(1);
		}
		
		FD<Dynamic,T>(const FD<Dynamic,T>& f):
		v(f.v),g(NULL),nVars(0){
			changeGradientSize(f.nVars);
			std::copy(&f.g[0],&f.g[0]+nVars,&g[0]);
		}
		
#if __cplusplus >= 201103L
		FD<Dynamic,T>(FD<Dynamic,T>&& f):
		v(f.v),g(f.g),nVars(f.nVars){
			//report that we've stolen the contents of f
			f.g=NULL;
			f.nVars=0;
		}
#endif
		
		FD<Dynamic,T>& operator=(const FD<Dynamic,T>& f){
			if(&f!=this){
				v=f.v;
				changeGradientSize(f.nVars);
				std::copy(&f.g[0],&f.g[0]+nVars,&g[0]);
			}
			return(*this);
		}
		
#if __cplusplus >= 201103L
		FD<Dynamic,T>& operator=(FD<Dynamic,T>&& f){
			if(&f!=this){
				v=f.v;
				nVars=f.nVars;
				g=f.g;
				//report that we've stolen the contents of f
				f.g=NULL;
				f.nVars=0;
			}
			return(*this);
		}
#endif
		
		FD<Dynamic,T>& operator=(const T& t){
			v=t;
			std::fill(&g[0],&g[0]+nVars,0);
			return(*this);
		}
		
		~FD<Dynamic,T>(){
			delete[] g;
		}
		
		void assignIndex(unsigned int index){
			changeGradientSize(index+1);
			assert(index<nVars);
			g[index]=T(1);
		}
		
		explicit
		operator T() const{
			return(v);
		}
		
		const T& value() const{
			return(v);
		}
		
		const T& derivative(unsigned int index) const{
			assert(index<nVars);
			return(g[index]);
		}
		
		void copyGradient(T grad[]) const{
			std::copy(&g[0],&g[0]+nVars,&grad[0]);
		}
		
		//unary +
		FD<Dynamic,T> operator+() const{
			return(*this);
		}
		
		//unary -
		FD<Dynamic,T> operator-() const{
			FD<Dynamic,T> r(*this);
			r.v=-r.v;
			r.changeGradientSize(nVars);
			std::transform(&r.g[0],&r.g[0]+nVars,&r.g[0],std::negate<T>());
			return(r);
		}
		
		//addition
		template <typename U>
		FD<Dynamic,T>& operator+=(const U& u){
			v+=T(u);
			return(*this);
		}
		
		FD<Dynamic,T>& operator+=(const FD<Dynamic,T>& f){
			v+=f.v;
			changeGradientSize(f.nVars);
			std::transform(&g[0],&g[0]+f.nVars,&f.g[0],&g[0],std::plus<T>());
			return(*this);
		}
		
		template <typename U>
		FD<Dynamic,T> operator+(const U& u) const{
			return(FD<Dynamic,T>(*this)+=u);
		}
		
		FD<Dynamic,T> operator+(const FD<Dynamic,T>& f) const{
			return(FD<Dynamic,T>(*this)+=f);
		}
		
		//subtraction
		template <typename U>
		FD<Dynamic,T>& operator-=(const U& u){
			v-=T(u);
			return(*this);
		}
		
		FD<Dynamic,T>& operator-=(const FD<Dynamic,T>& f){
			v-=f.v;
			changeGradientSize(f.nVars);
			std::transform(&g[0],&g[0]+f.nVars,&f.g[0],&g[0],std::minus<T>());
			return(*this);
		}
		
		template <typename U>
		FD<Dynamic,T> operator-(const U& u) const{
			return(FD<Dynamic,T>(*this)-=u);
		}
		
		FD<Dynamic,T> operator-(const FD<Dynamic,T>& f) const{
			return(FD<Dynamic,T>(*this)-=f);
		}
		
		//multiplication
		template <typename U>
		FD<Dynamic,T>& operator*=(const U& u){
			T t(u);
			v*=t;
			std::transform(&g[0],&g[0]+nVars,&g[0],std::bind2nd(std::multiplies<T>(),t));
			return(*this);
		}
		
		FD<Dynamic,T>& operator*=(const FD<Dynamic,T>& f){
			changeGradientSize(f.nVars);
			for(unsigned int i=0; i<f.nVars; i++)
				g[i] = g[i]*f.v + f.g[i]*v;
			v*=f.v;
			return(*this);
		}
		
		template<typename U>
		FD<Dynamic,T> operator*(const U& u) const{
			return(FD<Dynamic,T>(*this)*=u);
		}
		
		FD<Dynamic,T> operator*(const FD<Dynamic,T>& f) const{
			return(FD<Dynamic,T>(*this)*=f);
		}
		
		//division
		template<typename U>
		FD<Dynamic,T>& operator/=(const U& u){
			T t(u);
			v/=t;
			std::transform(&g[0],&g[0]+nVars,&g[0],std::bind2nd(std::divides<T>(),t));
			return(*this);
		}
		
		FD<Dynamic,T>& operator/=(const FD<Dynamic,T>& f){
			v/=f.v;
			changeGradientSize(f.nVars);
			for(unsigned int i=0; i<f.nVars; i++)
				g[i] = (g[i] - v*f.g[i])/f.v;
			return(*this);
		}
		
		template<typename U>
		FD<Dynamic,T> operator/(const U& u) const{
			return(FD<Dynamic,T>(*this)/=u);
		}
		
		FD<Dynamic,T> operator/(const FD<Dynamic,T>& f) const{
			return(FD<Dynamic,T>(*this)/=f);
		}
		
		template <template<int,class> class F, int D, class T_>
		friend struct detail::dimensionExtractor;
		template <int nVars_, typename T_, typename U>
		friend FD<nVars_,T_> operator/(const U& u, const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> cos(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> sin(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> tan(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> acos(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> asin(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> atan(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> cosh(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> sinh(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> tanh(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> exp(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> log(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> log10(const FD<nVars_,T_>& f);
		template <typename FD_t, typename U>
		friend FD_t pow(const FD_t& b, const U& e, typename boost::enable_if< boost::is_arithmetic< U >, int >::type);
		template <typename FD_t, typename U>
		friend FD_t pow(const U& b, const FD_t& e, typename boost::enable_if< boost::is_arithmetic< U >, int >::type);
		template <typename FD_t, typename U>
		friend FD_t pow(const U& b, const FD_t& e, typename boost::enable_if< boost::is_arithmetic< U >, int >::type);
		template <typename FD_t>
		friend FD_t pow(const FD_t& b, const FD_t& e);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> sqrt(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> floor(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> ceil(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> abs(const FD<nVars_,T_>& f);
		template <int nVars_, typename T_>
		friend FD<nVars_,T_> fabs(const FD<nVars_,T_>& f);
	};
	
//	}
	
	namespace std{
		//an autodiff::FD<nVars,T> has all the same properties as T
		template <int nVars, typename T>
		class numeric_limits</*autodiff::*/FD<nVars,T> >{
		public:
			static const bool is_specialized = numeric_limits<T>::is_specialized;
			
			static /*autodiff::*/FD<nVars,T> min() throw(){
				return(/*autodiff::*/FD<nVars,T>(numeric_limits<T>::min()));
			}
			static /*autodiff::*/FD<nVars,T> max() throw(){
				return(/*autodiff::*/FD<nVars,T>(numeric_limits<T>::max()));
			}
			
			static const int digits = numeric_limits<T>::digits;
			static const int digits10 = numeric_limits<T>::digits10;
			static const bool is_signed = numeric_limits<T>::is_signed;
			static const bool is_integer = numeric_limits<T>::is_integer;
			static const bool is_exact = numeric_limits<T>::is_exact;
			static const int radix = numeric_limits<T>::radix;
			
			static /*autodiff::*/FD<nVars,T> epsilon() throw(){
				return(/*autodiff::*/FD<nVars,T>(numeric_limits<T>::epsilon()));
			}
			static /*autodiff::*/FD<nVars,T> round_error() throw(){
				return(/*autodiff::*/FD<nVars,T>(numeric_limits<T>::round_error()));
			}
			
			static const int min_exponent = numeric_limits<T>::min_exponent;
			static const int min_exponent10 = numeric_limits<T>::min_exponent10;
			static const int max_exponent = numeric_limits<T>::max_exponent;
			static const int max_exponent10 = numeric_limits<T>::max_exponent10;
			
			static const bool has_infinity = numeric_limits<T>::has_infinity;
			static const bool has_quiet_NaN = numeric_limits<T>::has_quiet_NaN;
			static const bool has_signaling_NaN = numeric_limits<T>::has_signaling_NaN;
			static const float_denorm_style has_denorm = numeric_limits<T>::has_denorm;
			static const bool has_denorm_loss = numeric_limits<T>::has_denorm_loss;
			
			static /*autodiff::*/FD<nVars,T> infinity() throw(){
				return(/*autodiff::*/FD<nVars,T>(numeric_limits<T>::infinity()));
			}
			static /*autodiff::*/FD<nVars,T> quiet_NaN() throw(){
				return(/*autodiff::*/FD<nVars,T>(numeric_limits<T>::quiet_NaN()));
			}
			static /*autodiff::*/FD<nVars,T> signaling_NaN() throw(){
				return(/*autodiff::*/FD<nVars,T>(numeric_limits<T>::signaling_NaN()));
			}
			static /*autodiff::*/FD<nVars,T> denorm_min() throw(){
				return(/*autodiff::*/FD<nVars,T>(numeric_limits<T>::denorm_min()));
			}
			
			static const bool is_iec559 = numeric_limits<T>::is_iec559;
			static const bool is_bounded = numeric_limits<T>::is_bounded;
			static const bool is_modulo = numeric_limits<T>::is_modulo;
			static const bool traps = numeric_limits<T>::traps;
			static const bool tinyness_before = numeric_limits<T>::tinyness_before;
			static const float_round_style round_style = numeric_limits<T>::round_style;
		};
		
		template <int nVars, typename T>
		bool isnan(const /*autodiff::*/FD<nVars,T>& f){
			return(isnan((T)f));
		}
	}
	
#endif //AUTODIFF_H_INCLUDED