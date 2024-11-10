SOURCES := $(wildcard src/*/main.cu)
GLOBAL_UTILS := $(wildcard src/utils/*.c)

OUTPUTS := $(SOURCES:src/%/main.cu=out/%.o)

all: out $(OUTPUTS)

out:
	mkdir -p $@

clean:
	rm -f out/*

.SECONDEXPANSION:

$(OUTPUTS): out/%.o: $(GLOBAL_UTILS) $$(wildcard src/%/utils/*.c) src/%/main.cu
	nvcc $^ -o $@

