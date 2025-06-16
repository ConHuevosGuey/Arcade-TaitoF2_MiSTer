CC = m68k-elf-gcc
OBJCOPY = m68k-elf-objcopy
OBJDUMP = m68k-elf-objdump
MAME = ../util/taito_mame
PYTHON = uv run
SPLIT_ROM = $(PYTHON) ../util/split_rom.py
VASM = ../util/vasmz80_oldstyle
MISTER_HOSTNAME=mister-dev


TARGET = driftout_test
SRCS = init.c main.c interrupts_default.c comms.c tilemap.c input.c color.c printf/printf.c
SRCS += obj_test.c audio_240p.c tc0360pri.c
CHR = font.chr 
ASM = savestate.s

BUILD_DIR = build/$(TARGET)
ROM_DIR = ../roms

OBJS = $(addprefix $(BUILD_DIR)/, $(SRCS:c=o)) \
	   $(addprefix $(BUILD_DIR)/, $(CHR:chr=o)) \
	   $(addprefix $(BUILD_DIR)/, $(ASM:s=o))
BUILD_DIRS = $(sort $(dir $(OBJS))) 
GLOBAL_DEPS = Makefile

DEFINES = -DPRINTF_SUPPORT_DECIMAL_SPECIFIERS=0 \
	-DPRINTF_SUPPORT_EXPONENTIAL_SPECIFIERS=0 \
	-DPRINTF_SUPPORT_LONG_LONG=0 \
	-DPRINTF_ALIAS_STANDARD_FUNCTION_NAMES=1 \
	-DPRINTF_ALIAS_STANDARD_FUNCTION_NAMES_HARD=1

CFLAGS = -march=68000 -ffreestanding $(DEFINES) -O2 --std=c2x -g
LIBS = -lgcc
LDFLAGS = -march=68000 -static -nostdlib -g


ifeq ($(TARGET),finalb_test)
GAME = finalb
CPU_ROM_LOW   = b82-09.ic23
CPU_ROM_HIGH  = b82-17.ic11
AUDIO_CPU_ROM = b82_10.ic5
CPU_ROM_SIZE  = 0x40000
DEFINES += -DGAME_FINALB=1
else ifeq ($(TARGET),ssi_test)
GAME = ssi
CPU_ROM_LOW   = c64_15-1.ic9 
CPU_ROM_HIGH  = c64_16-1.ic8
AUDIO_CPU_ROM = c64-09.13
CPU_ROM_SIZE  = 0x40000
DEFINES += -DGAME_SSI=1
else ifeq ($(TARGET),qjinsei_test)
GAME = qjinsei
CPU_ROM_LOW   = d48-09 
CPU_ROM_HIGH  = d48-10
AUDIO_CPU_ROM = d48-11 
CPU_ROM_SIZE  = 0x40000
DEFINES += -DGAME_QJINSEI=1
else ifeq ($(TARGET),driftout_test)
GAME = driftout
CPU_ROM_LOW   = ic46.rom
CPU_ROM_HIGH  = ic45.rom
AUDIO_CPU_ROM = do_50.rom
CPU_ROM_SIZE  = 0x40000
DEFINES += -DGAME_DRIFTOUT=1
else
error
endif

GAME_DIR = $(BUILD_DIR)/$(GAME)
BUILT_BINS = $(addprefix $(GAME_DIR)/, $(CPU_ROM_LOW) $(CPU_ROM_HIGH) $(AUDIO_CPU_ROM))

ROMPATH = ../$(BUILD_DIR);../$(ROM_DIR)

all: $(BUILT_BINS)

$(BUILD_DIR)/cpu.bin: $(BUILD_DIR)/cpu.elf
	$(OBJCOPY) -O binary $< $@

$(GAME_DIR)/$(CPU_ROM_HIGH): $(BUILD_DIR)/cpu.bin | $(GAME_DIR)
	$(SPLIT_ROM) $@ $< 2 1 $(CPU_ROM_SIZE)

$(GAME_DIR)/$(CPU_ROM_LOW): $(BUILD_DIR)/cpu.bin | $(GAME_DIR)
	$(SPLIT_ROM) $@ $< 2 0 $(CPU_ROM_SIZE)

$(GAME_DIR)/$(AUDIO_CPU_ROM): artemio/sound.asm | $(GAME_DIR)
	$(VASM) -Fbin -nosym -chklabels -o $@ $<


$(BUILD_DIR)/%.o: %.c $(GLOBAL_DEPS) | $(BUILD_DIRS)
	@echo $@
	@$(CC) -MMD -o $@ $(CFLAGS) -c $<

$(BUILD_DIR)/%.o: %.s $(GLOBAL_DEPS) | $(BUILD_DIRS)
	@echo $@
	@$(CC) -MMD -o $@ $(CFLAGS) -c $<

$(BUILD_DIR)/%.o: %.chr $(GLOBAL_DEPS) | $(BUILD_DIRS)
	$(OBJCOPY) -I binary -O default $< $@

$(BUILD_DIR)/cpu.elf: $(OBJS)
	@echo $@
	@$(CC) -T $(GAME).ld -o $@ $(LDFLAGS) $^ $(LIBS)
	@$(OBJDUMP) -d -S $@ > $(BUILD_DIR)/cpu.asm

$(BUILD_DIRS):
	mkdir -p $@

$(GAME_DIR):
	mkdir -p $@

$(BUILD_DIR)/$(TARGET).zip: $(BUILT_BINS)
	zip -j $(BUILD_DIR)/$(TARGET).zip $^



debug: $(BUILT_BINS)
	mkdir -p mame
	cd mame && ../$(MAME) -window -nomaximize -resolution0 640x480 -debug -debugscript ../debugscript.txt -rompath "$(ROMPATH)" $(GAME)

run: $(BUILT_BINS)
	mkdir -p mame
	cd mame && ../$(MAME) -window -nomaximize -resolution0 640x480 -rompath "$(ROMPATH)" $(GAME)

picorom: $(BUILT_BINS)
	picorom reset cpu_low low
	picorom upload cpu_low $(GAME_DIR)/$(CPU_ROM_LOW)
	picorom upload cpu_high $(GAME_DIR)/$(CPU_ROM_HIGH)
	#picorom upload audio_cpu $(GAME_DIR)/$(AUDIO_CPU_ROM) 1mbit
	picorom reset cpu_low z

mister: $(BUILD_DIR)/$(TARGET).zip

.PHONY: run debug picorom mister

-include $(OBJS:o=d)
