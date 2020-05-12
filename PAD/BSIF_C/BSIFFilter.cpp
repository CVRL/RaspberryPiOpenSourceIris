//
//  BSIFFilter.cpp
//  TCLDetection

// Based on methods found at http://www.ee.oulu.fi/~jkannala/bsif/bsif.html


#include "BSIFFilter.hpp"
#include "filters.h"
#include <string>

BSIFFilter::BSIFFilter(void) {}

void BSIFFilter::loadFilter(int dimension, int bitlength, double** arr)
{
    int size = dimension; 
    int bits = bitlength;

    // Set the filter name
    std::stringstream nameStream;
    nameStream << "filter_" << size << "_" << size << "_" << bits;
    std::string filtername = nameStream.str();

    // load the hard-coded filters
    t_filtermap filters = build_filter_map();

    // Retrieve filter from filtermap (dereference pointer to pointer to change it)
    *arr = filters[filtername];
}







// convert linear indexing to subscript indexing
int s2i(int size, int bits, int row, int col, int bit)
{
    
    // C++ and python use row-major order, so the last dimension is contiguous
    // in doubt, refer to https://en.wikipedia.org/wiki/Row-_and_column-major_order#Column-major_order
    return bit + bits*(col+size*row);
    
}





// build a map containing all the hard-coded ICA Filters, used to load the filters
t_filtermap build_filter_map()
{
    
    // this is the map of filters
    t_filtermap filters;
    
    // we have to add all filters to the map
    // 3x3 filters
    std::string filterName = "filter_3_3_5";
    double* theFilter = &filter_3_3_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_3_3_6";
    theFilter = &filter_3_3_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_3_3_7";
    theFilter = &filter_3_3_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_3_3_8";
    theFilter = &filter_3_3_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 5x5 filters
    filterName = "filter_5_5_5";
    theFilter = &filter_5_5_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_6";
    theFilter = &filter_5_5_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_7";
    theFilter = &filter_5_5_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_8";
    theFilter = &filter_5_5_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_9";
    theFilter = &filter_5_5_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_10";
    theFilter = &filter_5_5_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_11";
    theFilter = &filter_5_5_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_5_5_12";
    theFilter = &filter_5_5_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 7x7 filters
    filterName = "filter_7_7_5";
    theFilter = &filter_7_7_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_6";
    theFilter = &filter_7_7_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_7";
    theFilter = &filter_7_7_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_8";
    theFilter = &filter_7_7_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_9";
    theFilter = &filter_7_7_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_10";
    theFilter = &filter_7_7_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_11";
    theFilter = &filter_7_7_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_7_7_12";
    theFilter = &filter_7_7_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 9x9 filters
    filterName = "filter_9_9_5";
    theFilter = &filter_9_9_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_6";
    theFilter = &filter_9_9_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_7";
    theFilter = &filter_9_9_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_8";
    theFilter = &filter_9_9_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_9";
    theFilter = &filter_9_9_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_10";
    theFilter = &filter_9_9_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_11";
    theFilter = &filter_9_9_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_9_9_12";
    theFilter = &filter_9_9_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 11x11 filters
    filterName = "filter_11_11_5";
    theFilter = &filter_11_11_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_6";
    theFilter = &filter_11_11_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_7";
    theFilter = &filter_11_11_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_8";
    theFilter = &filter_11_11_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_9";
    theFilter = &filter_11_11_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_10";
    theFilter = &filter_11_11_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_11";
    theFilter = &filter_11_11_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_11_11_12";
    theFilter = &filter_11_11_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 13x13 filters
    filterName = "filter_13_13_5";
    theFilter = &filter_13_13_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_6";
    theFilter = &filter_13_13_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_7";
    theFilter = &filter_13_13_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_8";
    theFilter = &filter_13_13_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_9";
    theFilter = &filter_13_13_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_10";
    theFilter = &filter_13_13_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_11";
    theFilter = &filter_13_13_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_13_13_12";
    theFilter = &filter_13_13_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 15x15 filters
    filterName = "filter_15_15_5";
    theFilter = &filter_15_15_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_6";
    theFilter = &filter_15_15_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_7";
    theFilter = &filter_15_15_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_8";
    theFilter = &filter_15_15_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_9";
    theFilter = &filter_15_15_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_10";
    theFilter = &filter_15_15_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_11";
    theFilter = &filter_15_15_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_15_15_12";
    theFilter = &filter_15_15_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    // 17x17 filters
    filterName = "filter_17_17_5";
    theFilter = &filter_17_17_5[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_6";
    theFilter = &filter_17_17_6[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_7";
    theFilter = &filter_17_17_7[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_8";
    theFilter = &filter_17_17_8[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_9";
    theFilter = &filter_17_17_9[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_10";
    theFilter = &filter_17_17_10[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_11";
    theFilter = &filter_17_17_11[0];
    filters.insert(t_filterpair(filterName, theFilter));
    filterName = "filter_17_17_12";
    theFilter = &filter_17_17_12[0];
    filters.insert(t_filterpair(filterName, theFilter));
    
    return filters;
    
}

