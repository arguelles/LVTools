#include "lv_search.h"

//===================MAIN======================================================//
//===================MAIN======================================================//

int main(int argc, char** argv)
{
    if(not (argc == 10)){
        std::cout << "Invalid number of arguments. The arguments should be given as follows: \n"
                     "1) Path to the effective area hdf5.\n"
                     "2) Path to the observed events file.\n"
                     "3) Path to Chris flux file with DOM efficiency correction.\n"
                     "4) Path to the kaon component flux.\n"
                     "5) Path to the pion component flux.\n"
                     "6) Path to the prompt component flux.\n"
                     "7-9) LLH parameters.\n"
                     << std::endl;
        exit(1);
    }

    // paths
    std::string effective_area_filename = std::string(argv[1]);
    std::string data_filename = std::string(argv[2]);
    std::string chris_flux_filename = std::string(argv[3]); // also contains DOM efficiency correction
    std::string kaon_filename = std::string(argv[4]);
    std::string pion_filename = std::string(argv[5]);
    std::string prompt_filename = std::string(argv[6]);

    LVSearch lv_search(effective_area_filename,data_filename,chris_flux_filename,kaon_filename,pion_filename,prompt_filename);

    std::array<double, 3> osc_params = {atof(argv[7]),atof(argv[8]),atof(argv[9])};
    std::cout << lv_search.llh(osc_params).likelihood << std::endl;

    return 0;
}
