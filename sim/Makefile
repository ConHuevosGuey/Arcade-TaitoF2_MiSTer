
UNAME=$(shell uname -s)
ifeq ($(UNAME),Linux)
	OS := Linux
endif
ifeq ($(UNAME),Darwin)
	OS := OSX
endif

USE_AUTO_SS = 1
OBJ_DIR = obj
VERILATED_DIR = verilated

GAME ?=

VERILATOR = verilator
VERILATOR_ARGS = --cc --make gmake --trace-fst --Mdir $(VERILATED_DIR) -Ihdl --MMD --MP -Wno-TIMESCALEMOD
PYTHON = python3

VERILATOR_INC = $(shell pkg-config --variable=includedir verilator)
VERILATOR_CPP = verilated.cpp verilated_fst_c.cpp verilated_threads.cpp
VERILATOR_OBJS = $(patsubst %.cpp, $(OBJ_DIR)/verilator/%.o, $(VERILATOR_CPP))

SRCS = imgui/imgui.cpp \
		imgui/imgui_draw.cpp \
		imgui/imgui_tables.cpp \
		imgui/imgui_widgets.cpp \
		imgui/backends/imgui_impl_sdl2.cpp \
		imgui/backends/imgui_impl_sdlrenderer2.cpp \
		dis68k/dis68k.cpp \
		sim_state.cpp \
		sim.cpp \
		games.cpp \
		imgui_wrap.cpp \
		tc0200obj.cpp \
		tc0360pri.cpp \
		miniz.cpp \
		file_search.cpp

OBJS = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRCS))

DEPFLAGS = -MT $@ -MMD -MP -MF $(OBJ_DIR)/$*.d

CC=cc
CXX=c++
CPPFLAGS= --std=gnu++17 -g $(DEPFLAGS) -O2
CPPFLAGS+=-I$(VERILATED_DIR) $(shell pkg-config --cflags verilator)
CPPFLAGS+=$(shell pkg-config --cflags sdl2)
CPPFLAGS+=-Iimgui/ -Iimgui/backends/

LDFLAGS=-g

ifeq ($(OS),OSX)
LDFLAGS+=-Wl,-U,__Z15vl_time_stamp64v,-U,__Z13sc_time_stampv
endif

LDLIBS=$(shell pkg-config --libs sdl2)

HDL_SRC  = ../rtl/fx68k/hdl/verilator/fx68k.sv ../rtl/fx68k/hdl/verilator/fx68kAlu.sv ../rtl/fx68k/hdl/verilator/uaddrPla.sv
HDL_SRC += ../rtl/system_consts.sv ../rtl/address_translator.sv ../rtl/game_board_config.sv
HDL_SRC += ../rtl/rom_cache.sv
HDL_SRC += ../rtl/tmp82c265.sv ../rtl/tc0030cmd.sv ../rtl/te7750.sv
HDL_SRC += ../rtl/F2.sv
HDL_SRC += ../rtl/tc0430grw.sv
HDL_SRC += ../rtl/tc0100scn.sv ../rtl/tc0110pr.sv ../rtl/tc0220ioc.sv ../rtl/tc0200obj.sv ../rtl/tc0140syt.sv
HDL_SRC += ../rtl/tc0260dar.sv ../rtl/tc0200obj_extender.sv ../rtl/tc0360pri.sv ../rtl/tc0190fmc.sv
HDL_SRC += ../rtl/ram.sv ../rtl/jtframe_frac_cen.v ../rtl/memory_stream.sv ../rtl/savestates.sv ../rtl/ddram.sv
HDL_SRC += ../rtl/audio_mix.sv ../sys/iir_filter.v
HDL_SRC += verilator.vlt


ifeq ($(USE_AUTO_SS),1)
VERILATOR_ARGS += -DUSE_AUTO_SS
HDL_SRC += ../rtl/tv80_auto_ss.sv
HDL_SRC += ../rtl/jt10_auto_ss.sv
else
HDL_SRC += ../rtl/tv80/tv80s.v ../rtl/tv80/tv80_alu.v ../rtl/tv80/tv80_reg.v ../rtl/tv80/tv80_core.v ../rtl/tv80/tv80_mcode.v
VERILATOR_ARGS += -F ../rtl/jt12/ver/verilator/gather.f
endif

all: sim

$(VERILATED_DIR)/F2.mk: $(HDL_SRC) $(HDL_GEN) Makefile
	$(VERILATOR) $(VERILATOR_ARGS) -o F2 --prefix F2 --top F2 $(HDL_SRC)

$(VERILATED_DIR)/F2__ALL.a: $(VERILATED_DIR)/F2.mk $(HDL_SRC) $(HDL_GEN)
	$(MAKE) -C $(VERILATED_DIR) -f F2.mk

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) -o $@ -c $< $(CPPFLAGS)

$(OBJ_DIR)/verilator/%.o: $(VERILATOR_INC)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -o $@ -c $< $(CPPFLAGS)

$(OBJS): $(VERILATED_DIR)/F2__ALL.a

microrom.mem: ../rtl/fx68k/hdl/microrom.mem
	cp $< $@

nanorom.mem: ../rtl/fx68k/hdl/nanorom.mem
	cp $< $@

sim: $(OBJS) $(VERILATOR_OBJS) $(VERILATED_DIR)/F2__ALL.a | microrom.mem nanorom.mem
	$(CXX) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS) -lpthread -lz

run: sim
	./sim $(GAME)

.PHONY: clean all run

clean:
	rm -r $(OBJ_DIR) $(VERILATED_DIR)

DEPFILES := $(SRCS:%.cpp=$(OBJ_DIR)/%.d)
$(DEPFILES):
-include $(wildcard $(DEPFILES))
