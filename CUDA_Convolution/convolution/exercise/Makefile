include ../../cuda.mk

convolution: convolution.o
	$(LINK.cu) $^ $(CULOADLIBES) $(CULDLIBS) -o $@ -O2 -L/usr/X11R6/lib -lm -lpthread -lX11

clean:
	$(RM) convolution.o convolution
