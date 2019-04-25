CXX=$(CROSS_COMPILE)g++
RM=rm

ALG=3
G_KERNEL=3

SRC=../../src/
3DPARTY=../../3dparty/
CXXSOURCES=$(wildcard ./*.cpp) $(wildcard $(SRC)/*.cpp)

OPENCV_LIB=/usr/local/opt/opencv3
OPENCV_LIB2=/usr/local

INCLUDE = -I$(OPENCV_LIB)/include -I$(OPENCV_LIB2)/include -I$(SRC) -I$(3DPARTY) -I.

CXXFLAGS+= $(INCLUDE) -O2 -Wall -Wextra -pedantic -ansi -std=c++11 -DALG=$(ALG) -DG_KERNEL=$(G_KERNEL) 
# remove this for production code
CXXFLAGS+= -DDEBUG
LDFLAGS+= -L$(OPENCV_LIB)/lib -L. -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_photo 

# We will need to compile differently for different architectures
# on Mac OSX -static-libstdc++ is ignored so we don't add it
ifneq ($(shell uname), Darwin)
    LDFLAGS += -static-libstdc++
    CXXFLAGS += -s
endif

CXXOBJECTS=$(CXXSOURCES:.cpp=.o)
OBJECTS=$(CXXOBJECTS)

C_OUT_NAME=main.exe

all: $(SOURCES) $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(C_OUT_NAME) $(LDFLAGS)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	-$(RM) -f $(C_OUT_NAME) $(OBJECTS) STDOUT.txt app.log stat_info_data.csv
