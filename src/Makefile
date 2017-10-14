CC = g++
CPPFLAGS = -c -g

DEPS = facenet_tf.h
SOURCES = classify.cpp facenet_tf.cpp
CPPFLAGS += -I/usr/lib/python2.7/site-packages/tensorflow/include/external/nsync/public/ -I../include `pkg-config --cflags opencv`
LDFLAGS += -L../lib -ltensorflow_cc -ltensorflow_framework `pkg-config --libs opencv`

OBJ = $(SOURCES:.cpp=.o)

%.o : %.c $(DEPS)
	$(CC) -std=c++11 -c -o $@ $< $(CPPFLAGS)

classify: $(OBJ)
	$(CC) -std=c++11 -o $@ $^ $(LDFLAGS)

clean:
	rm *.o classify
	
