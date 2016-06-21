#ifndef LV_GSL_INTEGRATE_H
#define LV_GSL_INTEGRATE_H

class IntegrateWorkspace {
private:
public:
    gsl_integration_workspace* ws;
    IntegrateWorkspace(size_t limit) {
        ws=gsl_integration_workspace_alloc(limit);
    }
    ~IntegrateWorkspace() {
        gsl_integration_workspace_free(ws);
    }
};

template<typename FunctionType>
double integrate(IntegrateWorkspace& ws, FunctionType f, double a, double b, double acc=1e-7, unsigned int max_iter=5000){
    double (*wrapper)(double,void*)=[](double x, void* params){
        FunctionType& f=*static_cast<FunctionType*>(params);
        return(f(x));
    };

    double result, error;
    gsl_function F;
    F.function = wrapper;
    F.params = &f;

    gsl_integration_qags(&F, a, b, 0, acc, max_iter, ws.ws, &result, &error);

    return(result);
}

template<typename FunctionType>
double integrate(FunctionType f, double a, double b, double acc=1e-7, unsigned int max_iter=5000){
    IntegrateWorkspace ws(5000);
    return integrate(ws, a, b, acc, max_iter);
}

#endif
