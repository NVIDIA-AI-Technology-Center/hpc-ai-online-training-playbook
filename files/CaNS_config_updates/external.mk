#
# external libraries compilation
#
ifeq ($(strip $(GPU)),1)
libs: $(wildcard $(LIBS_DIR)/2decomp-fft/src/*.f90)
	cd $(LIBS_DIR)/2decomp-fft && make
	#cd $(LIBS_DIR)/cuDecomp && make lib -j
	cd $(LIBS_DIR)/cuDecomp && mkdir build && cd build && \
	cmake -DCUDECOMP_ENABLE_NVSHMEM=ON .. && make

libsclean: $(wildcard $(LIBS_DIR)/2decomp-fft/src/*.f90)
	cd $(LIBS_DIR)/2decomp-fft && make clean
	cd $(LIBS_DIR)/cuDecomp && make clean
else
libs: $(wildcard $(LIBS_DIR)/2decomp-fft/src/*.f90)
	cd $(LIBS_DIR)/2decomp-fft && make
libsclean: $(wildcard $(LIBS_DIR)/2decomp-fft/src/*.f90)
	cd $(LIBS_DIR)/2decomp-fft && make clean
endif
