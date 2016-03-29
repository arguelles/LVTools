#ifndef LBFGSB_INTERFACE_H
#define LBFGSB_INTERFACE_H

class BFGS_FunctionBase{
public:
	///Evaluate the value of the function at x
	virtual double evalF(std::vector<double> x)=0;
	///Evaluate the value of the function and its gradient at x
	virtual std::pair<double,std::vector<double>> evalFG(std::vector<double> x)=0;
};

class LBFGSB_Driver{
private:
	//variables related to the full set of function parameters specified by the user
	
	///number of variables in the problem being solved
	size_t nVar;
	///initial values for the parameters
	std::vector<double> parameterValues;
	///scaling factors for the parameters
	std::vector<double> parameterScales;
	///the lower bounds for each variable
	std::vector<double> lowerBounds;
	///the upper bounds for each variable
	std::vector<double> upperBounds;
	///defines which bounds are used for each variable
	/// 0 : variable i is unbounded
	/// 1 : variable i is bounded below only
	/// 2 : variable i is bounded below and above
	/// 3 : variable i is bounded above only
	std::vector<int> boundsTypes;
	//indexes of fixed parameters; must be kept in sorted order
	std::vector<size_t> fixedParams;
	
	//variables related to the minimizer itself
	
	///number of previous gradients to store for the hessian approximation
	size_t historySize;
	///tolerance for absolute change in function values; if the change in
	///value between iterations is smaller than this value, the minimization
	///is considered to have converged
	double changeTol;
	///tolerance for magnitude of the gradient; if after an iteration the
	///largest component of the scaled function gradient (the gradient
	///divided element-wise by parameterScales) the minimization is considered
	///to have converged
	double gradTol;
	
	///records the value of the objective function at the best point found by minimizing
	///not valid until after minimize() has been called at least once!
	double finalFunctionValue;
	///Number of function evaluations performed during the last call to minimize()
	size_t nEvals;
	
	///copy values from a vector with all parameters
	///to a vector including only free parameters
	///\param externalVector the evaulation position including all parameters
	///\param internalVector the evaulation position including only free parameters
	///\pre internalVector.size()==externalVector.size()-fixedParams.size()
	template<typename T>
	void externalToInternal(const std::vector<T>& externalVector,
							std::vector<T>& internalVector){
		//i is the external index
		//j is the internal index
		//k is the index in fixedParams
		const size_t nFixed=fixedParams.size();
		for(size_t i=0, j=0, k=0; i<nVar; i++){
			if(k<nFixed && fixedParams[k]==i){
				k++;
				continue;
			}
			internalVector[j]=externalVector[i];
			j++;
		}
	}
	
	///copy values from a vector with only free parameters
	///to a vector including all parameters, without disturbing the
	///existing values in the full vector corresponding to fixed parameters
	///\param internalVector the evaulation position including only free parameters
	///\param externalVector the evaulation position including all parameters
	///\pre internalVector.size()==externalVector.size()-fixedParams.size()
	template<typename T>
	void internalToExternal(const std::vector<T>& internalVector,
							std::vector<T>& externalVector){
		//i is the external index
		//j is the internal index
		//k is the index in fixedParams
		const size_t nFixed=fixedParams.size();
		for(size_t i=0, j=0, k=0; i<nVar; i++){
			if(k<nFixed && fixedParams[k]==i){
				k++;
				continue;
			}
			externalVector[i]=internalVector[j];
			j++;
		}
	}
	
public:
	LBFGSB_Driver():
	nVar(0),
	historySize(10),
	changeTol(1e-8),
	gradTol(1e-8),
	finalFunctionValue(std::numeric_limits<double>::quiet_NaN()),
	nEvals(0)
	{}
	
	///Set the number of past gradient values which should be used to estimate the hessian
	void setHistorySize(unsigned int size){
		historySize=size;
	}
	
	///Set the termination tolerance threshold for changes in successive values of the objective function
	void setChangeTolerance(double tol){
		changeTol=tol;
	}
	///Set the termination tolerance threshold for the maximum component of the scaled gradient
	void setGradientTolerance(double tol){
		gradTol=tol;
	}
	
	///Add a function parameter
	///\param value the initial value to use for this parameter
	///\param scale the factor by which the component of the gradient
	///             corresponding to this parameter should be rescaled
	///             when testing for the flatness of the objective function
	///\param lowerBound the smallest value allowed for this parameter;
	///             if negative infinity this parameter is not bounded below
	///\param upperBound the largest value allowed for this parameter;
	///             if infinity this parameter is not bounded above
	void addParameter(double value, double scale=1,
					  double lowerBound=-std::numeric_limits<double>::infinity(),
					  double upperBound=std::numeric_limits<double>::infinity()){
		nVar++;
		parameterValues.push_back(value);
		parameterScales.push_back(scale);
		lowerBounds.push_back(lowerBound);
		upperBounds.push_back(upperBound);
		int boundType=0;
		if(lowerBound>-std::numeric_limits<double>::infinity())
			boundType=1;
		if(upperBound<std::numeric_limits<double>::infinity())
			boundType=(boundType==1?2:3);
		boundsTypes.push_back(boundType);
		if(boundType==2 && upperBound<=lowerBound)
			throw std::runtime_error("Conflicting parameter bounds");
	}
	///Set the initial value for a parameter
	///\param idx the index of the parameter to set
	///\param val the value to set for the parameter
	void setParameterValue(size_t idx, double val){
		assert(idx<nVar);
		parameterValues[idx]=val;
	}
	///Get the value of a parameter
	///\param idx the index of the parameter to fetch
	double getParameterValue(size_t idx){
		assert(idx<nVar);
		return(parameterValues[idx]);
	}
	///Set the gradient scaling factor for a parameter
	///\param idx the index of the parameter to set
	///\param scale the scale factor to use for this parameter's component of the gradient
	void setParameterScale(size_t idx, double scale){
		assert(idx<nVar);
		parameterScales[idx]=scale;
	}
	///Set or unset the lower limit for a parameter
	///\param idx the index of the parameter to set
	///\param lim the limiting value to set;
	///           if negative infinity any lower bound on this parameter is removed
	void setParameterLowerLimit(size_t idx, double lim){
		assert(idx<nVar);
		lowerBounds[idx]=lim;
		if(lim>-std::numeric_limits<double>::infinity()){
			if(boundsTypes[idx]==0)
				boundsTypes[idx]=1;
			else if(boundsTypes[idx]==3)
				boundsTypes[idx]=2;
		}
		else{ //we are _unsetting_ this bound!
			if(boundsTypes[idx]==1)
				boundsTypes[idx]=0;
			else if(boundsTypes[idx]==2)
				boundsTypes[idx]=3;
		}
	}
	///Set or unset the upper limit for a parameter
	///\param idx the index of the parameter to set
	///\param lim the limiting value to set;
	///           if infinity any upper bound on this parameter is removed
	void setParameterUpperLimit(size_t idx, double lim){
		assert(idx<nVar);
		upperBounds[idx]=lim;
		if(lim<std::numeric_limits<double>::infinity()){
			if(boundsTypes[idx]==0)
				boundsTypes[idx]=3;
			else if(boundsTypes[idx]==1)
				boundsTypes[idx]=2;
		}
		else{ //we are _unsetting_ this bound!
			if(boundsTypes[idx]==3)
				boundsTypes[idx]=0;
			else if(boundsTypes[idx]==1)
				boundsTypes[idx]=2;
		}
	}
	///Mark a parameter to be held fixed
	///All fixed parameters will be hidden from the minimizer,
	///and their initial values will be used for all function evaluations.
	///Has no effect if the parameter is already fixed.
	///\param idx the index of the parameter to fix
	void fixParameter(size_t idx){
		if(std::find(fixedParams.begin(),fixedParams.end(),idx)==fixedParams.end()){
			if(std::binary_search(fixedParams.begin(),fixedParams.end(),idx))
				return;
			fixedParams.push_back(idx);
			std::sort(fixedParams.begin(),fixedParams.end());
		}
	}
	///Release a fixed parameter to be minimized normally
	///Has no effect if the parameter is already free.
	///\param idx the index of the parameter to free
	void freeParameter(size_t idx){
		auto it=std::find(fixedParams.begin(),fixedParams.end(),idx);
		if(it!=fixedParams.end())
			fixedParams.erase(it);
	}
	
	///Perform a minimization of a function with respect to the free parameters
	///\param func the objective function to be minimized
	///\returns true if minimization successfully converged
	///\post the values of all free parameters will be modified to match the location of the best point
	///the minimizer was able to find (even if the minimization was not deemed to have converged), and
	///can be retrieved by GetParameterValue() or minimumPosition(), and the function value at that point
	///can be retrieved using minimumValue(). Additionally, numberOfEvaluations() can report the number of
	///function evaluations performed during the minimization attempt.
	bool minimize(BFGS_FunctionBase&& func){
		//number of currently free variables
		int effectiveNVar=nVar-fixedParams.size();
		if(effectiveNVar==0){ //the 0D case is so easy we don't have to call a library
			finalFunctionValue=func.evalF(parameterValues);
			nEvals=1;
			return(true);
		}
		assert(effectiveNVar>0);
		
		///buffer for receiving instructions from setulb
		char task[60];
		///buffer for internal calculations (iwa)
		std::vector<int> internalBuf(3*effectiveNVar);
		///buffer for internal gradient history calculations (wa)
		std::vector<double> internalGradBuf((2*historySize + 5)*effectiveNVar + 11*historySize*historySize + 8*historySize);
		///buffer for internal string manipulation
		char textBuf[60];
		///buffer for integer diagnostics
		int diagnosticBuf[48];
		///buffer for floating point diagnostics
		double diagnosticBuf2[29];
		int iprint=0; //dummy var
		int histSize=historySize;
		
		//position known to the minimizer, contains only free variables
		std::vector<double> pos(effectiveNVar);
		double fval;
		//complete gradient with all variables
		std::vector<double> fullGrad;
		//gradient known to the minimizer, contains only free variables
		std::vector<double> grad(effectiveNVar);
		//lower bounds for free variables only
		std::vector<double> lb(effectiveNVar);
		externalToInternal(lowerBounds,lb);
		//upper bounds for free variables only
		std::vector<double> ub(effectiveNVar);
		externalToInternal(upperBounds,ub);
		//bounds types for free variables only
		std::vector<int> bt(effectiveNVar);
		externalToInternal(boundsTypes,bt);
		//parameter scales for free variables only
		std::vector<double> scales(effectiveNVar);
		externalToInternal(parameterScales,scales);
		
		
		//prepare for first iteration
		memset(task, 0, 60);
		strncpy(task, "START", 5);
		externalToInternal(parameterValues,pos);
		nEvals=0;
		
		bool done=false;
		
		while(!done){
			setulb_(&effectiveNVar,     //n
					&histSize,          //m
					&pos[0],            //x
					&lb[0],             //l
					&ub[0],             //u
					&bt[0],             //nbd
					&scales[0],         //scale
					&fval,              //f
					&grad[0],           //g
					&changeTol,         //factr
					&gradTol,           //pgtol
					&internalGradBuf[0],//wa
					&internalBuf[0],    //iwa
					&task[0],           //task
					&iprint,            //iprint, unused
					&textBuf[0],        //csave
					&diagnosticBuf[0],  //lsave
					&diagnosticBuf[4],  //isave
					&diagnosticBuf2[0]  //dsave
					);
			
			switch(task[0]){
				case 'A': //failure
					//copy back fit point and value
					internalToExternal(pos,parameterValues);
					finalFunctionValue=fval;
					done=true;
					//failure indicated to the user by return value below
					break;
				case 'C': //converged
					//copy back fit point and value
					internalToExternal(pos,parameterValues);
					finalFunctionValue=fval;
					done=true;
					break;
				case 'E':
					throw std::runtime_error("Invalid parameters passed to setulb");
				case 'F': //need to evaluate function
					internalToExternal(pos,parameterValues);
					std::tie(fval,fullGrad)=func.evalFG(parameterValues);
					externalToInternal(fullGrad,grad);
					nEvals++;
					break;
				case 'N': //iteration completed
					//check iteration count/number of evaluations?
					break;
			}
		}
		return(task[0]=='C');
	}
	
	///Get the smallest function value found by the minimizer
	///\pre minimize() must have been called previously
	double minimumValue() const{
		return(finalFunctionValue);
	}
	
	///Get the parametervalues at which the smallest function value
	///was found by the minimizer
	///\pre minimize() must have been called previously
	std::vector<double> minimumPosition() const{
		return(parameterValues);
	}
	
	///Get the number of function (and gradient) evaluations
	///performed during the last minimization attempt
	///\pre minimize() must have been called previously
	size_t numberOfEvaluations() const{
		return(nEvals);
	}
};

#endif