#ifndef LV_GSL_INTEGRATE_H
#define LV_GSL_INTEGRATE_H

template<typename FunctionType>
double integrate(FunctionType f, double a, double b){
    double (*wrapper)(double,void*)=[](double x, void* params){
        FunctionType& f=*static_cast<FunctionType*>(params);
        return(f(x));
    };

    gsl_integration_workspace* ws=gsl_integration_workspace_alloc(5000);
    double result, error;
    gsl_function F;
    F.function = wrapper;
    F.params = &f;

    gsl_integration_qags(&F, a, b, 0, 1e-7, 5000, ws, &result, &error);
    gsl_integration_workspace_free(ws);

    return(result);
}

#endif
