//
//  BSIFFilter.hpp
//  TCLDetection



#ifndef BSIFFilter_hpp
#define BSIFFilter_hpp

#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <iostream>
#include <sstream>



class BSIFFilter
{
public:
    BSIFFilter();
    
    void loadFilter(int dimension, int bitlength, double** arr);
    
    std::string filtername;
    std::string downFiltername;
private:
    int size;
    int bits;
    double* myFilter;
};

int s2i(int size, int bits, int i, int j, int k);


// types to represent a map of filters and pairs to add to the map
typedef std::map<std::string, double*> t_filtermap;
typedef std::pair<std::string, double*> t_filterpair;

t_filtermap build_filter_map();

#endif /* BSIFFilter_hpp */
