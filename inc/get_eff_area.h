#include <hdf5.h>
#include <nuSQuIDS/marray.h>
#include <iostream>

//If you active this then this should be a stand alone program to test the effective area
//Thanks gabriel for this piece of code. Abrazos Carlos.
//#define TEST_EFFECTIVE_AREA

enum PTypes {
	NUMU = 0,
	NUMUBAR,
	NUTAU,
	NUTAUBAR
};

enum Year {
	y2010 = 0,
	y2011
};

using AreaArray = std::array<std::array<nusquids::marray<double, 3>, 4>, 2>;

using AreaEdges = std::array<std::array<std::array<nusquids::marray<double, 1>, 3>, 4>, 2>;

nusquids::marray<double, 1> get_edge(std::string path, size_t idx, hid_t file) {
	size_t sizes[3] = {281, 12, 51};
	
	std::string real_path = path + "/bin_edges_" + std::to_string(idx);
	hid_t h5_edge = H5Dopen(file, real_path.c_str(), H5P_DEFAULT);
	
	double* buffer = new double[sizes[idx]];
	herr_t status = H5Dread(h5_edge, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
	if (status<0)
		throw std::runtime_error("Error reading dataset: " + real_path);
	
	nusquids::marray<double, 1> result{sizes[idx]};
	for (size_t i=0; i<sizes[idx]; i++)
		result[i] = buffer[i];
	
	delete[] buffer;
	H5Dclose(h5_edge);
	return result;
}

nusquids::marray<double, 3> get_data(std::string year, std::string particle, hid_t file, std::array<nusquids::marray<double, 1>, 3>& edges) {
	std::string path = year + "/" + particle;
	std::string real_path = path + "/area";
	hid_t h5_matrix = H5Dopen(file, real_path.c_str(), H5P_DEFAULT);
	
	double* buffer = new double[50*11*280];
	herr_t status = H5Dread(h5_matrix, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
	if (status<0)
		throw std::runtime_error("Error reading dataset: " + real_path);
	
	nusquids::marray<double, 3> result{280, 11, 50};
	for (size_t i=0; i < 280; i++)
		for (size_t j=0; j < 11; j++)
			for (size_t k=0; k < 50; k++)
				result[i][j][k] = buffer[k + 50*j + 50*11*i];
	
	delete[] buffer;
	H5Dclose(h5_matrix);
	
	for (size_t idx=0; idx<3; idx++)
		edges[idx] = get_edge(path, idx, file);
	
	return result;
}

AreaArray get_areas(std::string h5_filename, AreaEdges& edges) {
	hid_t h5_file = H5Fopen(h5_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	
	AreaArray result;
	
	std::string year_names[2];
	year_names[Year::y2010] = "2010"; year_names[Year::y2011] = "2011";
	std::string p_names[4];
	p_names[PTypes::NUMU] = "nu_mu";   p_names[PTypes::NUMUBAR] = "nu_mu_bar";
	p_names[PTypes::NUTAU] = "nu_tau"; p_names[PTypes::NUTAUBAR] = "nu_tau_bar";
	
	for (size_t i=0; i<2; i++)
		for (size_t j=0; j<4; j++) {
			result[i][j] = get_data(year_names[i], p_names[j], h5_file, edges[i][j]);
		}
	
	H5Fclose(h5_file);
	return result;
}

#ifdef TEST_EFFECTIVE_AREA

int main(int argc, char** argv) {
	AreaEdges edges;
	AreaArray areas = get_areas(std::string(argv[1]), edges);
	std::cout << areas[y2010][NUMUBAR][279][10][49] << std::endl;
	std::cout << areas[y2010][NUTAU][279][10][49] << std::endl;
	std::cout << areas[y2011][NUMU][279][10][49] << std::endl;
	std::cout << areas[y2011][NUTAUBAR][279][10][49] << std::endl;
	
	std::cout << areas[y2010][NUMUBAR][150][5][30] << std::endl; // 2.0138101901128178
	std::cout << areas[y2010][NUTAU][235][10][38] << std::endl; // 5367.9245359445686
	std::cout << areas[y2011][NUMU][150][5][30] << std::endl; //1.9013554586901076
	std::cout << areas[y2011][NUTAUBAR][215][9][38] << std::endl; // 1839.7605049077802
	
	std::cout << edges[y2010][NUMU][0][0] << std::endl; // 100
	std::cout << edges[y2010][NUMUBAR][1][5] << std::endl; // -0.5
	std::cout << edges[y2011][NUTAU][2][30] << std::endl; // 100000.0
	std::cout << edges[y2011][NUTAUBAR][0][150] << std::endl; // 562341.32519034913
}

#endif
