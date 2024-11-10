SOURCES := $(wildcard src/*/main.cu)
GLOBAL_C_UTILS := $(wildcard src/utils/*.c)
GLOBAL_CUDA_UTILS := $(GLOBAL_C_UTILS:.c=.cu)
OUTPUTS := $(SOURCES:src/%/main.cu=out/%.o)

all: out $(OUTPUTS)

out:
	mkdir -p $@

clean:
	rm -f out/*

.SECONDEXPANSION:

LOCAL_C_UTILS := $$(wildcard src/%/utils/*.c)
LOCAL_CUDA_UTILS := $(LOCAL_C_UTILS:.c=.cu)

$(OUTPUTS): out/%.o: $(GLOBAL_C_UTILS) $(LOCAL_C_UTILS) $(LOCAL_CUDA_UTILS) src/%/main.cu
	nvcc $^ -o $@

