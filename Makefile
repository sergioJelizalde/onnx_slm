# Makefile — build infer_arm64.c for ARM64 Bluefield 3 or native x86
#
# Usage:
#   make                  → native build (x86 for testing)
#   make arm64            → cross-compile for AArch64 (requires cross toolchain)
#   make arm64_native     → native build ON the Bluefield 3 itself
#   make clean

# ── detect ORT install location ───────────────────────────────────────────────
# Option 1: from pip wheel (Python-installed ORT)
#   python3 -c "import onnxruntime as ort; print(ort.__file__)"
#   → /usr/local/lib/python3.10/dist-packages/onnxruntime/__init__.py
#   Headers are at: site-packages/onnxruntime/capi/include/
#   Libs at:        site-packages/onnxruntime/capi/
#
# Option 2: from official ORT release tarball
#   https://github.com/microsoft/onnxruntime/releases
#   → onnxruntime-linux-aarch64-<ver>.tgz
#   Set ORT_ROOT to extracted directory.

# ── auto-detect ORT from Python install ────────────────────────────────────
ORT_PYROOT := $(shell python3 -c \
    "import onnxruntime as o, os; \
     p=os.path.dirname(o.__file__)+'/capi'; \
     print(p if os.path.isdir(p) else '')" 2>/dev/null)

ifneq ($(ORT_PYROOT),)
    ORT_INC := $(ORT_PYROOT)/include
    ORT_LIB := $(ORT_PYROOT)
else
    # fallback: system install
    ORT_INC := /usr/local/include/onnxruntime
    ORT_LIB := /usr/local/lib
endif

CC_NATIVE := gcc
CC_CROSS  := aarch64-linux-gnu-gcc

CFLAGS := -O2 -Wall -std=c11 \
          -I$(ORT_INC) \
          -L$(ORT_LIB) \
          -DORT_API_VERSION=18
LIBS   := -lonnxruntime -lm

# ── targets ────────────────────────────────────────────────────────────────────
.PHONY: all native arm64 arm64_native clean info

all: native

native: infer_arm64
	@echo "✓ Native binary: ./infer_arm64"

infer_arm64: infer_arm64.c
	$(CC_NATIVE) $(CFLAGS) -o $@ $< $(LIBS) -Wl,-rpath,$(ORT_LIB)
	@echo "  ORT include: $(ORT_INC)"
	@echo "  ORT lib:     $(ORT_LIB)"

# Cross-compile for Bluefield 3 (run this on x86 dev machine)
# Requires: sudo apt install gcc-aarch64-linux-gnu
# Requires: ORT aarch64 prebuilt from:
#   https://github.com/microsoft/onnxruntime/releases
#   e.g. onnxruntime-linux-aarch64-1.20.1.tgz → set ORT_ARM64_ROOT
arm64: infer_arm64.c
ifndef ORT_ARM64_ROOT
	$(error ORT_ARM64_ROOT is not set. \
	  Download https://github.com/microsoft/onnxruntime/releases \
	  onnxruntime-linux-aarch64-*.tgz, extract it, and set ORT_ARM64_ROOT=<path>)
endif
	$(CC_CROSS) -O2 -Wall -std=c11 \
	    -I$(ORT_ARM64_ROOT)/include \
	    -L$(ORT_ARM64_ROOT)/lib \
	    -o infer_arm64_aarch64 infer_arm64.c \
	    -lonnxruntime -lm \
	    -Wl,-rpath,/usr/local/lib
	@echo "✓ AArch64 binary: ./infer_arm64_aarch64"
	@echo "  Copy to Bluefield 3 along with:"
	@echo "    $(ORT_ARM64_ROOT)/lib/libonnxruntime.so"
	@echo "    models/minilm_int8.onnx"
	@echo "    inputs/domains.bin"

# Build natively on the Bluefield 3 (run this ON the BF3)
arm64_native: infer_arm64.c
	gcc -O2 -Wall -std=c11 \
	    -I$(ORT_INC) -L$(ORT_LIB) \
	    -march=armv8.2-a+dotprod \
	    -o infer_arm64 infer_arm64.c \
	    -lonnxruntime -lm \
	    -Wl,-rpath,$(ORT_LIB)
	@echo "✓ Native AArch64 build"

info:
	@echo "ORT_INC = $(ORT_INC)"
	@echo "ORT_LIB = $(ORT_LIB)"
	@ls $(ORT_INC)/onnxruntime_c_api.h 2>/dev/null && \
	    echo "  onnxruntime_c_api.h ✓" || echo "  onnxruntime_c_api.h NOT FOUND"
	@ls $(ORT_LIB)/libonnxruntime* 2>/dev/null | head -3 || \
	    echo "  libonnxruntime not found"

clean:
	rm -f infer_arm64 infer_arm64_aarch64
