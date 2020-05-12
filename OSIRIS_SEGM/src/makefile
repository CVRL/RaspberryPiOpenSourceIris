PKG_CONFIG_PATH=/usr/lib/pkgconfig
export PKG_CONFIG_PATH
INCLUDES=-I/usr/include/opencv2

all : OsiMain.cpp OsiManager.cpp OsiEye.cpp OsiProcessings.cpp OsiCircle.cpp
	g++ OsiMain.cpp OsiManager.cpp OsiEye.cpp OsiProcessings.cpp OsiCircle.cpp -o osiris `pkg-config opencv --cflags --libs` -I/usr/lib/gcc/arm-linux-gnueabihf/6/include	
clean : osiris
	rm *[~o]
	
