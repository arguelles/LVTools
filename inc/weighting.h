/*
Infrastructure for building and combining objects to easily compute event weights

Supposing we have a type representing physics events:

struct event{
	...
	double zenithAngle;
	double energy;
	...
};

If we want to weight events inversely proportionally to the solid angle on the sky at the the 
zenith angle at which they are observed we can create a simple weighter type:

struct solidAngleWeighter : public GenericWeighter<solidAngleWeighter>{
	using result_type=double;
	result_type operator()(const event& e){
		return(1/cos(e.zenithAngle));
	}
};

Important points to note are that the weighter should derive publicly from GenericWeighter with 
itself as a template parameter (this enables interoperation with overloaded arithmetic operators 
for weighters), and that the weighter should define a member type called `result_type` which is the 
type of weight which it returns. 

The latter is important for weighters which weight events depending on one or more physics 
parameters, as we may want to use them with more complex types than just floating point numbers, to 
allow computing derivatives of weights with respect to the physics parameters. For example:

template<typename T>
struct powerlawWeighter : public GenericWeighter<powerlawWeighter<T>>{
private:
	T index;
public:
	using result_type=T;
	powerlawWeighter(T i):index(i){}
	result_type operator()(const event& e){
		return(pow(e.energy,index));
	}
};

We can now instantiate powerlawWeighters which do or don't compute the derivatives of weights with 
respect to the index of the powerlaw spectrum to which the weighting is performed. 

Overloaded arithmetic operations which are supported for weighters are addition, multiplication, 
and multiplication by scalars. So, we can construct a composite weighter:

double normalization=...;
auto myWeight=solidAngleWeighter()+normalization*powerlawWeighter<double>();

Arithmetic is also defined for working with references to weighters so that complex weighter objects 
need not be copied. Supposing we have some weighter type ExpensiveWeighter which we wish to avoid 
copying because it contains a large amount of internal state (like a lookup table of data loaded 
from a file), but we want to construct many different combined weighters based on an instance of 
ExpensiveWeighter, we can do something like the following:

ExpensiveWeighter e;
for(...){
	double normalization;
	//compute a new normalization
	auto w=normalization*std::cref(e);
	//use w to weight events
}

This will avoid copying e on every iteration of the loop. This technique comes with the usual 
limitations of reference semantics, however: The result of an expression using references must 
never be kept or used outside the sope of the objects to which the references refer! In particular, 
one must NOT do the following:

some_type makeAWeighter(){
	double factor=5;
	ExpensiveWeighter e;
	return(factor*cref(e)); //DANGER! The using the result of this function will give undefined behavior!
}

Instead, in situations like this one should just do the obvious thing and handle the weighter by 
value, and in general it is best to handle all weighters by value unless there is some good reason 
not to. 
*/

#ifndef LF_WEIGHTING_H
#define LF_WEIGHTING_H

#include <functional> //for std::reference_wrapper
#include <type_traits> //for result_of, enable_if, and is_base_of

//instead of being a base class in the usual sense this is used as a tag so that templated 
//overloaded arithmetic operators can be defined which will work for any weighers (anything 
//derived from an instantiation of GenericWeighter) but not other, unintended types. 
template<typename Derived>
struct GenericWeighter{
public:
	using derived_type=Derived;
	operator Derived&(){
		return(*static_cast<Derived*>(this));
	}
	operator const Derived&() const{
		return(*static_cast<const Derived*>(this));
	}
};

//a trivial weighter which just holds (and returns) a constant
//this type is useful mostly for modifying the results of other, more complex weighters via the 
//overloaded arithmetic operators (which generate it automatically for expressions involving
//constants)
template <typename T=double>
struct constantWeighter : public GenericWeighter<constantWeighter<T>>{
private:
	T c;
public:
	using result_type=T;
	constantWeighter(T w):c(w){}
	
	template<typename Event>
	T operator()(const Event& e) const{ return(c); }
};

namespace detail{

	struct adder{ //Hiss...
		template <typename T, typename U>
		auto operator()(T t, U u) const->decltype(t+u){
			return(t+u);
		}
		double neutral() const{
			return(0);
		}
	};
	
	struct multiplier{
		template <typename T, typename U>
		auto operator()(T t, U u) const->decltype(t*u){
			return(t*u);
		}
		double neutral() const{
			return(1);
		}
	};
	
	template <typename Folder, typename T, typename... Others>
	struct fold_traits;
	
	template <typename Folder, typename T>
	struct fold_traits<Folder,T>{
		using type=T;
	};
	
	template <typename Folder, typename T, typename... Others>
	struct fold_traits{
		using type=typename std::result_of<Folder(T,typename fold_traits<Folder,Others...>::type)>::type;
	};
	
	//this type more-or-less represents the higher order 'fold' function (foldr, specifically)
	template<typename Combiner, typename... WeighterTypes>
	struct combinedWeighterType : public GenericWeighter<combinedWeighterType<Combiner,WeighterTypes...>>{
	private:
		Combiner c;
		std::tuple<WeighterTypes...> weighters;
		
		template<unsigned int Index, typename Event>
		auto getSingleWeight(const Event& e) const->
		typename std::remove_reference<decltype(std::get<Index>(weighters))>::type::result_type{
			return(std::get<Index>(weighters)(e));
		}
		
		template<unsigned int Index, typename Event, typename Dummy, typename... MoreDummies>
		auto evalImpl(const Event& e, Dummy* d=nullptr) const->
		typename fold_traits<Combiner, typename Dummy::result_type, typename MoreDummies::result_type...>::type
		{
			return(c(getSingleWeight<Index,Event>(e),evalImpl<Index+1,Event,MoreDummies...>(e)));
		}
		
		template<unsigned int Index, typename Event>
		double evalImpl(const Event& e) const{
			return(c.neutral());
		}
	public:
		combinedWeighterType(Combiner c, WeighterTypes... ws):c(c),weighters(ws...){}
		
		using result_type=typename fold_traits<Combiner, typename WeighterTypes::result_type...>::type;
		
		template<typename Event>
		result_type operator()(const Event& e) const{
			return(evalImpl<0,Event,WeighterTypes...>(e));
		}
	};

}

//basic helper function for combining weighters
template<typename Combiner, typename... WeighterTypes>
detail::combinedWeighterType<Combiner,WeighterTypes...> combineWeighters(WeighterTypes... ws){
	return(detail::combinedWeighterType<Combiner,WeighterTypes...>(Combiner(),ws...));
}

// Overloaded operators for combining weighters

//sum of weighters
template<typename WT1, typename WT2>
detail::combinedWeighterType<detail::adder,WT1,WT2>
operator+(const GenericWeighter<WT1>& w1, const GenericWeighter<WT2>& w2){
	return(combineWeighters<detail::adder>(static_cast<const WT1>(w1),static_cast<const WT2>(w2)));
}

//product of weighters
template<typename WT1, typename WT2>
detail::combinedWeighterType<detail::multiplier,WT1,WT2>
operator*(const GenericWeighter<WT1>& w1, const GenericWeighter<WT2>& w2){
	return(combineWeighters<detail::multiplier>(static_cast<const WT1>(w1),static_cast<const WT2>(w2)));
}

//sum of a constant with a weighter
template<typename T, typename WT, typename=typename std::enable_if<!std::is_base_of<GenericWeighter<T>,T>::value>::type>
detail::combinedWeighterType<detail::adder,constantWeighter<T>,WT>
operator+(const T& c, const GenericWeighter<WT>& w){
	return(combineWeighters<detail::adder>(constantWeighter<T>(c),static_cast<const WT>(w)));
}

//sum of a weighter with a constant
template<typename T, typename WT, typename=typename std::enable_if<!std::is_base_of<GenericWeighter<T>,T>::value>::type>
detail::combinedWeighterType<detail::adder,constantWeighter<T>,WT>
operator+(const GenericWeighter<WT>& w, const T& c){
	return(combineWeighters<detail::adder>(constantWeighter<T>(c),static_cast<const WT>(w)));
}

//product of a constant with a weighter
template<typename T, typename WT, typename=typename std::enable_if<!std::is_base_of<GenericWeighter<T>,T>::value>::type>
detail::combinedWeighterType<detail::multiplier,constantWeighter<T>,WT>
operator*(const T& c, const GenericWeighter<WT>& w){
	return(combineWeighters<detail::multiplier>(constantWeighter<T>(c),static_cast<const WT>(w)));
}

//product of a weighter with a constant
template<typename T, typename WT, typename=typename std::enable_if<!std::is_base_of<GenericWeighter<T>,T>::value>::type>
detail::combinedWeighterType<detail::multiplier,constantWeighter<T>,WT>
operator*(const GenericWeighter<WT>& w, const T& c){
	return(combineWeighters<detail::multiplier>(constantWeighter<T>(c),static_cast<const WT>(w)));
}

// Overloaded operators for combining references to weighters
// these versions allow combining const references to weighters created by std::cref

//sum of weighters
// first reference
template<typename WT1, typename WT2, 
         typename=typename std::enable_if<std::is_base_of<GenericWeighter<WT1>,WT1>::value
                                          && std::is_base_of<GenericWeighter<WT2>,WT2>::value>::type>
detail::combinedWeighterType<detail::adder,std::reference_wrapper<const WT1>,WT2>
operator+(std::reference_wrapper<const WT1> w1, const GenericWeighter<WT2>& w2){
	return(combineWeighters<detail::adder>(w1,static_cast<const WT2>(w2)));
}
// second reference
template<typename WT1, typename WT2, 
         typename=typename std::enable_if<std::is_base_of<GenericWeighter<WT1>,WT1>::value
                                          && std::is_base_of<GenericWeighter<WT2>,WT2>::value>::type>
detail::combinedWeighterType<detail::adder,WT1,std::reference_wrapper<const WT2>>
operator+(const GenericWeighter<WT1>& w1, std::reference_wrapper<const WT2> w2){
	return(combineWeighters<detail::adder>(static_cast<const WT1>(w1),w2));
}
// both references
template<typename WT1, typename WT2, 
         typename=typename std::enable_if<std::is_base_of<GenericWeighter<WT1>,WT1>::value
                                          && std::is_base_of<GenericWeighter<WT2>,WT2>::value>::type>
detail::combinedWeighterType<detail::adder,std::reference_wrapper<const WT1>,std::reference_wrapper<const WT2>>
operator+(std::reference_wrapper<const WT1> w1, std::reference_wrapper<const WT2> w2){
	return(combineWeighters<detail::adder>(w1,w2));
}

//sum of a constant with a weighter
template<typename T, typename WT, typename=typename std::enable_if<std::is_base_of<GenericWeighter<WT>,WT>::value && !std::is_base_of<GenericWeighter<T>,T>::value>::type>
detail::combinedWeighterType<detail::adder,constantWeighter<T>,std::reference_wrapper<const WT>>
operator+(const T& c, std::reference_wrapper<const WT> w){
	return(combineWeighters<detail::adder>(constantWeighter<T>(c),w));
}

//sum of a weighter with a constant
template<typename T, typename WT, typename=typename std::enable_if<std::is_base_of<GenericWeighter<WT>,WT>::value && !std::is_base_of<GenericWeighter<T>,T>::value>::type>
detail::combinedWeighterType<detail::adder,constantWeighter<T>,std::reference_wrapper<const WT>>
operator+(std::reference_wrapper<const WT> w, const T& c){
	return(combineWeighters<detail::adder>(w,constantWeighter<T>(c)));
}

//product of weighters
// first reference
template<typename WT1, typename WT2, 
         typename=typename std::enable_if<std::is_base_of<GenericWeighter<WT1>,WT1>::value
                                          && std::is_base_of<GenericWeighter<WT2>,WT2>::value>::type>
detail::combinedWeighterType<detail::multiplier,std::reference_wrapper<const WT1>,WT2>
operator*(std::reference_wrapper<const WT1> w1, const GenericWeighter<WT2>& w2){
	return(combineWeighters<detail::multiplier>(w1,static_cast<const WT2>(w2)));
}
// second reference
template<typename WT1, typename WT2, 
         typename=typename std::enable_if<std::is_base_of<GenericWeighter<WT1>,WT1>::value
                                          && std::is_base_of<GenericWeighter<WT2>,WT2>::value>::type>
detail::combinedWeighterType<detail::multiplier,WT1,std::reference_wrapper<const WT2>>
operator*(const GenericWeighter<WT1>& w1, std::reference_wrapper<const WT2> w2){
	return(combineWeighters<detail::multiplier>(static_cast<const WT1>(w1),w2));
}
// both references
template<typename WT1, typename WT2, 
         typename=typename std::enable_if<std::is_base_of<GenericWeighter<WT1>,WT1>::value
                                          && std::is_base_of<GenericWeighter<WT2>,WT2>::value>::type>
detail::combinedWeighterType<detail::multiplier,std::reference_wrapper<const WT1>,std::reference_wrapper<const WT2>>
operator*(std::reference_wrapper<const WT1> w1, std::reference_wrapper<const WT2> w2){
	return(combineWeighters<detail::multiplier>(w1,w2));
}

//product of a constant with a weighter
template<typename T, typename WT, typename=typename std::enable_if<std::is_base_of<GenericWeighter<WT>,WT>::value && !std::is_base_of<GenericWeighter<T>,T>::value>::type>
detail::combinedWeighterType<detail::multiplier,constantWeighter<T>,std::reference_wrapper<const WT>>
operator*(const T& c, std::reference_wrapper<const WT> w){
	return(combineWeighters<detail::multiplier>(constantWeighter<T>(c),w));
}

//product of a weighter with a constant
template<typename T, typename WT, typename=typename std::enable_if<std::is_base_of<GenericWeighter<WT>,WT>::value && !std::is_base_of<GenericWeighter<T>,T>::value>::type>
detail::combinedWeighterType<detail::multiplier,constantWeighter<T>,std::reference_wrapper<const WT>>
operator*(std::reference_wrapper<const WT> w, const T& c){
	return(combineWeighters<detail::multiplier>(w,constantWeighter<T>(c)));
}

#endif