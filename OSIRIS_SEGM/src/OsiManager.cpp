/*******************************************************
* Open Source for Iris : OSIRIS
* Version : 4.0
* Date : 2011
* Author : Guillaume Sutra, Telecom SudParis, France
* License : BSD
********************************************************/

#include <fstream>
#include <iterator>
#include <stdexcept>
#include "OsiManager.h"
#include "OsiStringUtils.h"

using namespace std ;

namespace osiris
{


    // CONSTRUCTORS & DESTRUCTORS
    /////////////////////////////


    // Default constructor
    OsiManager::OsiManager ( )
    {
        // Associate lines of configuration file to the attributes
        mMapBool["Process segmentation"] = &mProcessSegmentation ;
        mMapString["List of images"] = &mFilenameListOfImages ;
        mMapString["Load original images"] = &mInputDirOriginalImages ;
        mMapString["Save segmented images"] = &mOutputDirSegmentedImages ;
        mMapString["Save contours parameters"] = &mOutputDirParameters ;
        mMapString["Save masks of iris"] = &mOutputDirMasks ;
        mMapInt["Minimum diameter for pupil"] = &mMinPupilDiameter ;
        mMapInt["Maximum diameter for pupil"] = &mMaxPupilDiameter ;
        mMapInt["Minimum diameter for iris"] = &mMinIrisDiameter ;
        mMapInt["Maximum diameter for iris"] = &mMaxIrisDiameter ;
        mMapString["Suffix for segmented images"] = &mSuffixSegmentedImages ;
        mMapString["Suffix for parameters"] = &mSuffixParameters ;
        mMapString["Suffix for masks of iris"] = &mSuffixMasks ;

        // Initialize all parameters
	// initConfiguration() ;        
    }





    // Default destructor
    OsiManager::~OsiManager ( )
    {
    }



    // OPERATORS
    ////////////



    // Initialize all configuration parameters
    void OsiManager::initConfiguration (const string & rFilename)
    {
        // Options of processing
        mProcessSegmentation = true;

        // Inputs
        mListOfImages.clear() ;
        mFilenameListOfImages = rFilename;
        mInputDirOriginalImages = "/" ;

        // Outputs
        mOutputDirSegmentedImages = "/home/pi/Desktop/iris/OSIRIS_SEGM/outputs/" ;
        mOutputDirParameters = "/home/pi/Desktop/iris/OSIRIS_SEGM/outputs/" ;
        mOutputDirMasks = "/home/pi/Desktop/iris/OSIRIS_SEGM/outputs/" ;

        // Parameters
        mMinPupilDiameter = 60 ;
        mMaxPupilDiameter = 140 ;
        mMinIrisDiameter = 160 ;
        mMaxIrisDiameter = 360 ;

        // Suffix for filenames
        mSuffixSegmentedImages = "_segm.bmp" ;
        mSuffixParameters = "_para.txt" ;
        mSuffixMasks = "_mask.bmp" ;

        // Load the list containing all images
        loadListOfImages() ;
    }



    // Load the application points from a textfile
    void OsiManager::loadListOfImages ( )
    {
        // Open the file
        ifstream file(mFilenameListOfImages.c_str(),ios::in) ;

        // If file is not opened
        if ( ! file )
        {
            throw runtime_error("Cannot load the list of images in " + mFilenameListOfImages) ;
        }

        // Fill in the list
        copy(istream_iterator<string>(file),istream_iterator<string>(),back_inserter(mListOfImages)) ;

        // Close the file
        file.close() ;

    } // end of function






    // Load, segment, normalize, encode, and save according to user configuration
    void OsiManager::processOneEye ( const string & rFileName , OsiEye & rEye )
    {
        //cout << "Process " << rFileName << endl ;

        // Strings handle
        OsiStringUtils osu ;

        // Get eye name
        string short_name = osu.extractFileName(rFileName) ;

        // Load original image only if segmentation is requested
        if ( mProcessSegmentation )
        {
            if ( mInputDirOriginalImages != "" )
            {
                rEye.loadOriginalImage(mInputDirOriginalImages+rFileName) ;                
            }
            else
            {
                throw runtime_error("Cannot segment/normalize without loading original image") ;
            }
        }

        /////////////////////////////////////////////////////////////////
        // SEGMENTATION : process, load
        /////////////////////////////////////////////////////////////////

        // Segmentation step
        if ( mProcessSegmentation )
        {
            rEye.segment(mMinIrisDiameter,mMinPupilDiameter,mMaxIrisDiameter,mMaxPupilDiameter) ;

            // Save segmented image
            /*
	    if ( mOutputDirSegmentedImages != "" )
            {
                rEye.saveSegmentedImage(mOutputDirSegmentedImages+short_name+mSuffixSegmentedImages) ;
            }
	    */
        }

        /////////////////////////////////////////////////////////////////
        // SAVE
        /////////////////////////////////////////////////////////////////

        // Save parameters
        if ( mOutputDirParameters != "" )
        {
            if ( !mProcessSegmentation )
            {
                cout << "Cannot save parameters because they are not computed" << endl ;
            }
            else
            {
                rEye.saveParameters(mOutputDirParameters+short_name+mSuffixParameters) ;
            }
        }

        // Save mask
        if ( mOutputDirMasks != "" )
        {
            if ( !mProcessSegmentation)
            {
                cout << "Cannot save masks because they are not computed" << endl ;
            }
            else
            {
                rEye.saveMask(mOutputDirMasks+short_name+mSuffixMasks) ;
            }
        }

    } // end of function


    // Run osiris
    void OsiManager::run ( )
    {
	/*
        cout << endl ;
        cout << "================" << endl ;
        cout << "Start processing" << endl ;
        cout << "================" << endl ;
        cout << endl ;
	*/

        for ( int i = 0 ; i < mListOfImages.size() ; i++ )
        {
            // Message on prompt command to know the progress
            //cout << i+1 << " / " << mListOfImages.size() << endl ;

            OsiEye eye ;
            processOneEye(mListOfImages[i],eye) ;

        } // end for images

	/*
        cout << endl ;
        cout << "==============" << endl ;
        cout << "End processing" << endl ;
        cout << "==============" << endl ;
        cout << endl ;
	*/

    } // end of function

} // end of namespace


