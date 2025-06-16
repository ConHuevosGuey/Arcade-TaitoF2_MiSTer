#include <stdint.h>
#include <stdbool.h>
#include "printf/printf.h"

#include "tc0100scn.h"
#include "tc0200obj.h"
#include "util.h"
#include "interrupts.h"
#include "tilemap.h"
#include "input.h"
#include "system.h"
#include "obj_test.h"
#include "color.h"

volatile uint32_t vblank_count = 0;
volatile uint32_t dma_count = 0;

void level5_handler()
{
    vblank_count++;
#if HAS_TC0360PRI
    tc0360pri_vblank();
#endif
    TC0220IOC->watchdog = 0;
}

void level6_handler()
{
    dma_count++;
}

uint32_t wait_vblank()
{
    uint32_t current = vblank_count;
    uint32_t count;
    while( current == vblank_count )
    {
        count++;
    }
    return count;
}

void wait_dma()
{
    uint32_t current = dma_count;
    while( current == dma_count )
    {
    }
}


uint8_t sine_wave[256] =
{
    0x80, 0x83, 0x86, 0x89, 0x8C, 0x90, 0x93, 0x96,
    0x99, 0x9C, 0x9F, 0xA2, 0xA5, 0xA8, 0xAB, 0xAE,
    0xB1, 0xB3, 0xB6, 0xB9, 0xBC, 0xBF, 0xC1, 0xC4,
    0xC7, 0xC9, 0xCC, 0xCE, 0xD1, 0xD3, 0xD5, 0xD8,
    0xDA, 0xDC, 0xDE, 0xE0, 0xE2, 0xE4, 0xE6, 0xE8,
    0xEA, 0xEB, 0xED, 0xEF, 0xF0, 0xF1, 0xF3, 0xF4,
    0xF5, 0xF6, 0xF8, 0xF9, 0xFA, 0xFA, 0xFB, 0xFC,
    0xFD, 0xFD, 0xFE, 0xFE, 0xFE, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFE, 0xFE, 0xFD,
    0xFD, 0xFC, 0xFB, 0xFA, 0xFA, 0xF9, 0xF8, 0xF6,
    0xF5, 0xF4, 0xF3, 0xF1, 0xF0, 0xEF, 0xED, 0xEB,
    0xEA, 0xE8, 0xE6, 0xE4, 0xE2, 0xE0, 0xDE, 0xDC,
    0xDA, 0xD8, 0xD5, 0xD3, 0xD1, 0xCE, 0xCC, 0xC9,
    0xC7, 0xC4, 0xC1, 0xBF, 0xBC, 0xB9, 0xB6, 0xB3,
    0xB1, 0xAE, 0xAB, 0xA8, 0xA5, 0xA2, 0x9F, 0x9C,
    0x99, 0x96, 0x93, 0x90, 0x8C, 0x89, 0x86, 0x83,
    0x80, 0x7D, 0x7A, 0x77, 0x74, 0x70, 0x6D, 0x6A,
    0x67, 0x64, 0x61, 0x5E, 0x5B, 0x58, 0x55, 0x52,
    0x4F, 0x4D, 0x4A, 0x47, 0x44, 0x41, 0x3F, 0x3C,
    0x39, 0x37, 0x34, 0x32, 0x2F, 0x2D, 0x2B, 0x28,
    0x26, 0x24, 0x22, 0x20, 0x1E, 0x1C, 0x1A, 0x18,
    0x16, 0x15, 0x13, 0x11, 0x10, 0x0F, 0x0D, 0x0C,
    0x0B, 0x0A, 0x08, 0x07, 0x06, 0x06, 0x05, 0x04,
    0x03, 0x03, 0x02, 0x02, 0x02, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x03,
    0x03, 0x04, 0x05, 0x06, 0x06, 0x07, 0x08, 0x0A,
    0x0B, 0x0C, 0x0D, 0x0F, 0x10, 0x11, 0x13, 0x15,
    0x16, 0x18, 0x1A, 0x1C, 0x1E, 0x20, 0x22, 0x24,
    0x26, 0x28, 0x2B, 0x2D, 0x2F, 0x32, 0x34, 0x37,
    0x39, 0x3C, 0x3F, 0x41, 0x44, 0x47, 0x4A, 0x4D,
    0x4F, 0x52, 0x55, 0x58, 0x5B, 0x5E, 0x61, 0x64,
    0x67, 0x6A, 0x6D, 0x70, 0x74, 0x77, 0x7A, 0x7D
};

extern char _binary_font_chr_start[];
extern char _binary_font_chr_end[];

#define NUM_SCREENS 10

static uint32_t frame_count;

void reset_screen()
{
    memset(TC0100SCN, 0, sizeof(TC0100SCN_Layout));
    int16_t base_x;
    int16_t base_y;
    uint16_t system_flags;

    bool flip = 0; //(input_dsw() & 0x02) == 0;

    if (flip)
    {
        base_x = 7;
        base_y = 16;
        system_flags = TC0100SCN_SYSTEM_FLIP;
    }
    else
    {
        base_x = 9;
        base_y = 0;
        system_flags = 0;
    }


    TC0100SCN_Ctrl->bg1_y = base_y;
    TC0100SCN_Ctrl->bg1_x = base_x;
    TC0100SCN_Ctrl->fg0_y = base_y;
    TC0100SCN_Ctrl->fg0_x = base_x;
    TC0100SCN_Ctrl->system_flags = system_flags;
#if 0 //GAME_DRIFTOUT
    TC0100SCN_Ctrl->layer_flags = TC0100SCN_LAYER_BG0_DISABLE | TC0100SCN_LAYER_BG1_DISABLE;
#else
    TC0100SCN_Ctrl->layer_flags = 0;
#endif    
    TC0100SCN_Ctrl->bg0_y = base_y;
    TC0100SCN_Ctrl->bg0_x = base_x;
    memcpy(TC0100SCN->fg0_gfx + ( 0x20 * 8 ), _binary_font_chr_start, _binary_font_chr_end - _binary_font_chr_start);

    for( int i = 0; i < 8; i++ )
    {
        TC0100SCN->fg0_gfx[8 + i] = 0xffff;
    }

    memsetw(TC0200OBJ, 0, 0x8000);

    set_default_palette();

#if HAS_TC0360PRI
    tc0360pri_set_obj_prio(5, 5, 7, 9);
    tc0360pri_set_tile_prio(2, 0, 0);
    tc0360pri_set_roz_prio(1, 1, 1, 1);
    tc0360pri_set_roz(0, 0);
#endif

#if HAS_TC0430GRW
    memsetw(TC0430GRW, 0, 64 * 64);

    TC0430GRW_Ctrl->origin_x = 0 << 12;
    TC0430GRW_Ctrl->origin_y = 8 << 12;
    TC0430GRW_Ctrl->dxx = 1 << 12;
    TC0430GRW_Ctrl->dxy = 0;
    TC0430GRW_Ctrl->dyx = 0;
    TC0430GRW_Ctrl->dyy = 1 << 12;
#endif
}

#if GAME_DRIFTOUT
#define SYM_BOX 0x20
#define SYM_SQUARE 0x20
#else
#define SYM_BOX 0x1
#define SYM_SQUARE 0x1b
#endif

void init_scn_general()
{
    reset_screen();

    frame_count = 0;

    // Max extent corner boundaries
    on_layer(BG0); pen_color(0);
    sym_at(1, 1, SYM_BOX);
    sym_at(1, 28, SYM_BOX);
    sym_at(40, 1, SYM_BOX);
    sym_at(40, 28, SYM_BOX);

    // Should not be visible
    pen_color(1);
    sym_at(0, 1, SYM_SQUARE);
    sym_at(1, 0, SYM_SQUARE);
    sym_at(0, 28, SYM_SQUARE);
    sym_at(1, 29, SYM_SQUARE);
    sym_at(40, 0, SYM_SQUARE);
    sym_at(41, 1, SYM_SQUARE);
    sym_at(40, 29, SYM_SQUARE);
    sym_at(41, 28, SYM_SQUARE);

    pen_color(6);
    print_at(4, 5, "LAYER BG0");
    on_layer(BG1);
    pen_color(3);
    print_at(4, 6, "LAYER BG1");

    on_layer(FG0);
    pen_color(0);
    print_at(4, 9, "The quick brown fox\njumps over the lazy dog.\n0123456789?/=-+*");


    on_layer(BG0);
    pen_color(9);
    sym_at(8, 14, 0x12);
    sym_at(8, 15, 0x12);
    sym_at(8, 16, 0x12);
    //print_at(8, 14, "ROW\nSCROLL\nBG0");
    print_at(8, 18, "-16 ROW\n-8 SCROLL\n 0 BG0\n 8 FIXED");

    on_layer(BG1);
    pen_color(2);
    print_at(19, 14, " COLSCROLL\n1222222222\n9012345678\n\n\nROWCOL\nSCROLL");

    on_layer(ROZ);
    pen_color(0);
    sym_at(2, 1, 0x67);
    sym_at(1, 2, 0x67);
    sym_at(40, 27, 0x67);
    sym_at(39, 28, 0x67);
}

void update_scn_general()
{
    wait_vblank();


    for( int y = 0; y < 24; y++ )
    {
//        TC0100SCN->bg0_rowscroll[14 * 8 + y] = sine_wave[(frame_count*2+(y*4)) & 0xff] >> 4;
    }

    for( int y = 0; y < 32; y++ )
    {
        //TC0100SCN->bg0_rowscroll[18 * 8 + y] = y + 1;
    }
        
/*    for( int x = 0; x < 6; x++ )
    {
        TC0100SCN->bg1_colscroll[20 + x] = sine_wave[(frame_count*2+(x*8)) & 0xff] >> 4;
    }*/
    for( int y = 0; y < 16; y++ )
    {
//        TC0100SCN->bg1_rowscroll[20 * 8 + y] = sine_wave[(frame_count*2+(y*4)) & 0xff] >> 4;
    }

    on_layer(FG0);
    pen_color(0);
    move_to(3, 2);
    print("VBL: %05X  FRAME: %05X\n", vblank_count, frame_count);
    print("%02X %02X %02X %02X\n",
          TC0220IOC->unk4,
          TC0220IOC->unk5,
          TC0220IOC->unk6,
          TC0220IOC->unk7
          );
    

    //TC0220IOC->coin_count |= 0xC0;
    if (frame_count % 100 == 0)
        TC0220IOC->coin_count ^= 0xC0;

    frame_count++;
}

int16_t invalid_read_count;

void init_scn_align()
{
    reset_screen();

    // Max extent corner boundaries
    on_layer(BG0); pen_color(0);
    sym_at(1, 1, 1);
    sym_at(1, 28, 1);
    sym_at(40, 1, 1);
    sym_at(40, 28, 1);

    // Should not be visible
    pen_color(1);
    sym_at(0, 1, 0x1b);
    sym_at(1, 0, 0x1b);
    sym_at(0, 28, 0x1b);
    sym_at(1, 29, 0x1b);
    sym_at(40, 0, 0x1b);
    sym_at(41, 1, 0x1b);
    sym_at(40, 29, 0x1b);
    sym_at(41, 28, 0x1b);

    pen_color(6);
    print_at(4, 3, "LAYER BG0");
    on_layer(BG1);
    pen_color(3);
    print_at(4, 4, "LAYER BG1");
    on_layer(FG0);
    pen_color(0);
    print_at(4, 5, "LAYER FG0");

    on_layer(BG0);
    pen_color(9);
    print_at(10, 8, "NO SCROLL");
    print_at(10, 9, "NO SCROLL");
    print_at(10, 20, "NO SCROLL");
    print_at(10, 21, "NO SCROLL");
    for (int y = 0; y < 10; y++)
    {
        sym_at(8, 10 + y, 0x12);
        print_at(10, 10 + y, "%d SCROLL", y);
    }

    for( int y = 0; y < 10 * 8; y++ )
    {
        TC0100SCN->bg0_rowscroll[(8 * 10) + y] = y - 39;
    }

    frame_count = 0;
    invalid_read_count = 0;
}

void update_scn_align()
{
    wait_vblank();

    bool flip = 0; //(input_dsw() & 0x02) == 0;

    int16_t base_x;
    int16_t base_y;
    uint16_t system_flags;

    if (flip)
    {
        base_x = 7;
        base_y = 16;
        system_flags = TC0100SCN_SYSTEM_FLIP;
    }
    else
    {
        base_x = 9;
        base_y = 0;
        system_flags = 0;
    }

    TC0100SCN_Ctrl->bg1_y = base_y;
    TC0100SCN_Ctrl->bg1_x = base_x;
    TC0100SCN_Ctrl->fg0_y = base_y;
    TC0100SCN_Ctrl->fg0_x = base_x;
    TC0100SCN_Ctrl->system_flags = system_flags;
    TC0100SCN_Ctrl->layer_flags = 0;
    TC0100SCN_Ctrl->bg0_y = base_y;
    TC0100SCN_Ctrl->bg0_x = base_x;

    TC0200OBJ_Inst *obj_ptr = TC0200OBJ;
    TC0200OBJ_Inst work;
    TC0200OBJ_Inst *o = &work;
    uint16_t cmd_base = OBJCMD_6BPP;

    if (flip) cmd_base |= OBJCMD_FLIPSCREEN;

    obj_reset(o);
    obj_cmd(o, cmd_base); obj_commit_reset(o, &obj_ptr);
    obj_master_xy(o, 100, 30); obj_commit_reset(o, &obj_ptr);
  
    GridOptions opt;
    opt.w = 2; opt.h = 2;
    opt.extra = opt.zoom = 0b100'000'000;
    opt.seq = 0b111'111'110;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b011'011'011;
    opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;
    opt.zoom_x = 0; opt.zoom_y = 0;
    opt.pos = 0;
    obj_grid(135, 40, &opt, &obj_ptr);

    opt.w = 1; opt.h = 1;
    opt.extra = opt.zoom = 1;
    opt.latch_x = opt.latch_y = 0;
    opt.seq = 0;
    opt.zoom_x = 0; opt.zoom_y = 0;
    opt.pos = 0;
    obj_grid(8, -15, &opt, &obj_ptr);
    obj_grid(8, 200, &opt, &obj_ptr);


    for( int i = 0; i < 512; i++ )
    {
        obj_ptr[i].pos0 = frame_count;
    }
    for( int i = 0; i < 512; i++ )
    {
        if( obj_ptr[i].pos0 != frame_count )
        {
            *(uint32_t *)0xff0000 = 1;
            invalid_read_count++;
        }
    }

    frame_count++;

    on_layer(BG0);
    move_to(20, 1);
    print("%u %u", frame_count, invalid_read_count);
}



uint16_t system_flags, layer_flags;

void init_scn_control_access()
{
    reset_screen();

    frame_count = 0;
    system_flags = 0;
    layer_flags = 0;

    const char *msg1 = "UP/DOWN ADJUST SYSTEM FLAGS";
    const char *msg2 = "LEFT/RIGHT ADJUST LAYER FLAGS";

    pen_color(0);
    on_layer(BG0);
    print_at(4, 20, msg1);
    on_layer(BG1);
    print_at(4, 20, msg1);
    on_layer(FG0);
    print_at(4, 20, msg1);

    on_layer(BG0);
    print_at(4, 21, msg2);
    on_layer(BG1);
    print_at(4, 21, msg2);
    on_layer(FG0);
    print_at(4, 21, msg2);
}

void update_scn_control_access()
{
    bool changed = false;

    if(input_pressed(LEFT))
    {
        if (system_flags == 0)
            system_flags = 1;
        else
            system_flags = system_flags << 1;
        changed = true;
    }

    if(input_pressed(RIGHT))
    {
        if (system_flags == 0)
            system_flags = 1 << 15;
        else
            system_flags = system_flags >> 1;
        changed = true;
    }

    if(input_pressed(UP))
    {
        if (layer_flags == 0)
            layer_flags = 1;
        else
            layer_flags = layer_flags << 1;
        changed = true;
    }

    if(input_pressed(DOWN))
    {
        if (layer_flags == 0)
            layer_flags = 1 << 15;
        else
            layer_flags = layer_flags >> 1;
        changed = true;
    }

    TC0100SCN_Ctrl->system_flags = system_flags;
    TC0100SCN_Ctrl->layer_flags = layer_flags;

    if (changed || frame_count == 0)
    {
        pen_color(0);
        on_layer(BG0);
        print_at(4, 4, "LAYER: %06X", layer_flags);
        on_layer(BG1);
        print_at(4, 5, "LAYER: %06X", layer_flags);
        on_layer(FG0);
        print_at(4, 6, "LAYER: %06X", layer_flags);

        pen_color(0);
        on_layer(BG0);
        print_at(20, 4, "SYSTEM: %06X", system_flags);
        on_layer(BG1);
        print_at(20, 5, "SYSTEM: %06X", system_flags);
        on_layer(FG0);
        print_at(20, 6, "SYSTEM: %06X", system_flags);
    }

    frame_count++;
}

void init_obj_general()
{
    reset_screen();

    frame_count = 0;
}

uint16_t obj_test_data[] =
{
    0x0000, 0x0000, 0x0000, 0x8000, 0x0000, 0x0300, 0x0000, 0x0000,
    0x0000, 0x0000, 0xA000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x50D0, 0x1058, 0x090C, 0x0000, 0x0000, 0x0000,
    0x13D3, 0x0000, 0x0000, 0x0000, 0x3D00, 0x0000, 0x0000, 0x0000,
    0x13D7, 0x0000, 0x0000, 0x0000, 0x3D00, 0x0000, 0x0000, 0x0000,
    0x13CF, 0x0000, 0x0000, 0x0000, 0xCD00, 0x0000, 0x0000, 0x0000,
    0x13D2, 0x0000, 0x0000, 0x0000, 0x7D00, 0x0000, 0x0000, 0x0000,
    0x13D6, 0x0000, 0x0000, 0x0000, 0x7D00, 0x0000, 0x0000, 0x0000,
    0x13CE, 0x0000, 0x0000, 0x0000, 0xCD00, 0x0000, 0x0000, 0x0000,
    0x13D1, 0x0000, 0x0000, 0x0000, 0x7D00, 0x0000, 0x0000, 0x0000,
    0x13D5, 0x0000, 0x0000, 0x0000, 0x7D00, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0xCD00, 0x0000, 0x0000, 0x0000,
    0x13D0, 0x0000, 0x0000, 0x0000, 0x7D00, 0x0000, 0x0000, 0x0000,
    0x13D4, 0x0000, 0x0000, 0x0000, 0x7500, 0x0000, 0x0000, 0x0000,
    0x0000, 0x1f1f, 0x5130, 0x0058, 0x090C, 0x0000, 0x0000, 0x0000,
    0x13D3, 0x0000, 0x0000, 0x0000, 0x3D00, 0x0000, 0x0000, 0x0000,
    0x13D7, 0x0000, 0x0000, 0x0000, 0x3D00, 0x0000, 0x0000, 0x0000,
    0x13CF, 0x0000, 0x0000, 0x0000, 0xCD00, 0x0000, 0x0000, 0x0000,
    0x13D2, 0x0000, 0x0000, 0x0000, 0x7D00, 0x0000, 0x0000, 0x0000,
    0x13D6, 0x0000, 0x0000, 0x0000, 0x7D00, 0x0000, 0x0000, 0x0000,
    0x13CE, 0x0000, 0x0000, 0x0000, 0xCD00, 0x0000, 0x0000, 0x0000,
    0x13D1, 0x0000, 0x0000, 0x0000, 0x7D00, 0x0000, 0x0000, 0x0000,
    0x13D5, 0x0000, 0x0000, 0x0000, 0x7D00, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0xCD00, 0x0000, 0x0000, 0x0000,
    0x13D0, 0x0000, 0x0000, 0x0000, 0x7D00, 0x0000, 0x0000, 0x0000,
    0x13D4, 0x0000, 0x0000, 0x0000, 0x7500, 0x0000, 0x0000, 0x0000,
};

void update_obj_general()
{
    wait_dma();
    memcpyw(TC0200OBJ, obj_test_data, sizeof(obj_test_data) / 2);
    wait_dma();

    // Max extent corner boundaries
    on_layer(BG0); pen_color(0);
    sym_at(1, 1, 1);
    sym_at(1, 28, 1);
    sym_at(40, 1, 1);
    sym_at(40, 28, 1);

    on_layer(BG0); pen_color(2);
    sym_at(17, 11, 1);
    sym_at(22, 13, 1);
    sym_at(17, 16, 1);
    sym_at(20, 16, 1);

    pen_color(1);
    sym_at(18, 15, 0x1b);


    // Should not be visible
    pen_color(1);
    sym_at(0, 1, 0x1b);
    sym_at(1, 0, 0x1b);
    sym_at(0, 28, 0x1b);
    sym_at(1, 29, 0x1b);
    sym_at(40, 0, 0x1b);
    sym_at(41, 1, 0x1b);
    sym_at(40, 29, 0x1b);
    sym_at(41, 28, 0x1b);


    frame_count++;
}


void send_sound_code(uint8_t code)
{
    *SYT_ADDR = 0;
    *SYT_DATA = (code >> 4) & 0xf;
    *SYT_ADDR = 1;
    *SYT_DATA = code & 0xf;
    *SYT_ADDR = 2;
    *SYT_DATA = (code >> 4) & 0xf;
    *SYT_ADDR = 3;
    *SYT_DATA = code & 0xf;
}

bool read_sound_response(uint8_t *code)
{
    uint8_t r = 0;
    *SYT_ADDR = 4;

    uint8_t status = *SYT_DATA;

    if (status & 0x4)
    {
        *SYT_ADDR = 0;
        r = (*SYT_DATA << 4);
        *SYT_ADDR = 1;
        r |= *SYT_DATA;

        if (code) *code = r;
        return true;
    }

    return false;
}

static uint8_t sound_code = 0;
static uint8_t sound_msg = 0;
static uint8_t sound_msg2 = 0;

static uint8_t sound_data[16];

void init_sound_test()
{
    reset_screen();
    
    // Reset
    *SYT_ADDR = 4;
    *SYT_DATA = 1;


    *SYT_ADDR = 4;
    *SYT_DATA = 0;


/*    wait_vblank();
    wait_vblank();
    wait_vblank();
    send_sound_code(0xFE);
    wait_vblank();
    wait_vblank();
    wait_vblank();
    wait_vblank();
    send_sound_code(0xA0);
    sound_code = 0xA0;*/
}

void update_sound_test()
{
    wait_vblank();

    if(input_pressed(LEFT))
    {
        sound_code--;
    }

    if(input_pressed(RIGHT))
    {
        sound_code++;
    }

    if(input_pressed(BTN1))
    {
        //send_sound_code(sound_code);
        extern void run_mdfourier();
        run_mdfourier();
    }
        
    *SYT_ADDR = 4;
    uint8_t status = (*SYT_DATA) & 0xf;

    if(status & 0x04)
    {
        *SYT_ADDR = 0;
        sound_msg = (*SYT_DATA) << 4;
        *SYT_ADDR = 1;
        sound_msg |= (*SYT_DATA) & 0x0f;
    }

    if(status & 0x08)
    {
        *SYT_ADDR = 2;
        sound_msg2 = (*SYT_DATA << 4);
        *SYT_ADDR = 3;
        sound_msg2 |= (*SYT_DATA & 0x0f);
    }

    if(input_pressed(BTN3))
    {
        pen_color(1);
        on_layer(FG0);
        move_to(4, 10);
        for( int i = 0; i < 16; i++ )
        {
            *SYT_ADDR = i;
            sound_data[i] = *SYT_DATA;
            print("%01X ", sound_data[i] & 0xf);
        }
    }

    pen_color(0);
    on_layer(FG0);
    print_at(4, 4, "SOUND CODE: %02X", sound_code);
    print_at(4, 5, "STATUS: %02X", status);
    print_at(4, 6, "MESSAGE: %02X %02X", sound_msg, sound_msg2);
}

void init_obj_test1()
{
    reset_screen();
}

void update_obj_test1()
{
    uint16_t cmd_base = OBJCMD_6BPP;
    wait_dma();

    TC0200OBJ_Inst *obj_ptr = TC0200OBJ;
    TC0200OBJ_Inst work;
    TC0200OBJ_Inst *o = &work;

    obj_reset(o);
    if (input_down(BTN1))
    {
        TC0200OBJ[4095].code = 0;
        obj_cmd(o, cmd_base);  obj_commit_reset(o, &obj_ptr);
    }
    else
    {   
        obj_cmd(o, cmd_base); obj_commit_reset(o, &obj_ptr);
    }
    obj_master_xy(o, 100, 30); obj_commit_reset(o, &obj_ptr);
  
    GridOptions opt;
    opt.w = 3; opt.h = 3;
    opt.extra = opt.zoom = 0b100'000'000;
    opt.zoom_x = 0; opt.zoom_y = 0;
    opt.pos = 0;

    // "Standard" way
    opt.seq = 0b111'111'110;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b011'011'011;
    opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(10, 10, &opt, &obj_ptr);

    // No sequence flags
    opt.seq = 0b000'000'000;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b011'011'011;
    opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(70, 10, &opt, &obj_ptr);

    // No inc y flags
    opt.seq = 0b000'000'000;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
    opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(130, 10, &opt, &obj_ptr);
    
    // Latch x always flags
    opt.seq = 0b000'000'000;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
    opt.latch_x = 0b011'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(190, 10, &opt, &obj_ptr);

    obj_master_xy(o, 100, 90); obj_commit_reset(o, &obj_ptr);

    // Now with zoom
    opt.zoom_x = 126; opt.zoom_y = 126;
    // "Standard" way
    opt.seq = 0b111'111'110;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b011'011'011;
    opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(10, 10, &opt, &obj_ptr);

    // No sequence flags
    opt.seq = 0b000'000'000;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b011'011'011;
    opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(70, 10, &opt, &obj_ptr);

    // No inc y flags
    opt.seq = 0b000'000'000;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
    opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(130, 10, &opt, &obj_ptr);

    // Latch x always flags
    opt.seq = 0b000'000'000;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
    opt.latch_x = 0b011'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(190, 10, &opt, &obj_ptr);
        
    obj_master_xy(o, 100, 150); obj_commit_reset(o, &obj_ptr);
    
    // Now with zoom and always set seq
    // No inc y flags
    opt.seq = 0b111'111'110;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
    opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(130, 10, &opt, &obj_ptr);

    // Latch x always flags
    opt.seq = 0b111'111'110;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
    opt.latch_x = 0b011'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(190, 10, &opt, &obj_ptr);
    
    
    obj_ptr = TC0200OBJ + 2049;
    obj_reset(o);
    obj_cmd(o, cmd_base); o->pos1 = 0x8000; obj_commit_reset(o, &obj_ptr);

}

void init_obj_test2()
{
    reset_screen();
}

void update_obj_test2()
{
    uint16_t cmd_base = OBJCMD_6BPP;
    wait_dma();

    TC0200OBJ_Inst *obj_ptr = TC0200OBJ;
    TC0200OBJ_Inst work;
    TC0200OBJ_Inst *o = &work;

    obj_reset(o);
    obj_cmd(o, cmd_base); obj_commit_reset(o, &obj_ptr);
    obj_master_xy(o, 100, 30); obj_commit_reset(o, &obj_ptr);

    GridOptions opt;
    opt.w = 3; opt.h = 3;
    opt.zoom_x = 0; opt.zoom_y = 0;
    opt.zoom = 0b100'000'000;
    opt.extra = 0b100'000'000;
    opt.pos = 0;

    opt.seq = 0b111'111'110;
    opt.latch_y = 0b011'111'111; opt.inc_y = 0b011'111'111;
    opt.latch_x = 0b011'000'100; opt.inc_x = 0b011'111'001;   
    obj_grid(10, 10, &opt, &obj_ptr);

    opt.seq = 0b000'000'000;
    opt.latch_y = 0b111'011'011; opt.inc_y = 0b111'011'011;
    opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(70, 10, &opt, &obj_ptr);

    obj_master_xy(o, 300, 30); obj_commit_reset(o, &obj_ptr);
    obj_extra_xy(o, 0, 0); obj_commit_reset(o, &obj_ptr);

    opt.extra = 0b010'000'000;
    opt.seq = 0b000'000'000;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
    opt.latch_x = 0b001'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(20, 20, &opt, &obj_ptr);
/*
    opt.seq = 0b000'000'000;
    opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
    opt.latch_x = 0b011'111'111; opt.inc_x = 0b000'100'100;   
    obj_grid(190, 10, &opt, &obj_ptr);
*/
}

int8_t test3_index;
void init_obj_test3()
{
    test3_index = 0;
    reset_screen();
}

#define IF_TEST(idx) if (idx == test3_index || test3_index == 0)
void update_obj_test3()
{
    uint16_t cmd_base = OBJCMD_6BPP;
    wait_dma();

    TC0200OBJ_Inst *obj_ptr = TC0200OBJ;
    TC0200OBJ_Inst work;
    TC0200OBJ_Inst *o = &work;

    obj_reset(o);
    obj_cmd(o, cmd_base); obj_commit_reset(o, &obj_ptr);
    obj_master_xy(o, 140, 70); obj_commit_reset(o, &obj_ptr);
    obj_extra_xy(o, -40, -40); obj_commit_reset(o, &obj_ptr);
 
    if (input_pressed(LEFT))
    {
        test3_index--;
    }

    if (input_pressed(RIGHT))
    {
        test3_index++;
    }

    if (test3_index > 10) test3_index = 0;
    if (test3_index < 0) test3_index = 10;

    GridOptions opt;
    opt.w = 3; opt.h = 3;
    opt.extra = 0;
    opt.pos = opt.zoom = 0b100'000'000;
    opt.zoom_x = 0; opt.zoom_y = 0;

    IF_TEST(1)
    {
        // "Standard" way
        opt.seq = 0b111'111'110;
        opt.latch_y = 0b011'011'011; opt.inc_y = 0b011'011'011;
        opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
        obj_grid(10, 10, &opt, &obj_ptr);
    }

    IF_TEST(2)
    {
        // No sequence flags
        opt.seq = 0b000'000'000;
        opt.latch_y = 0b011'011'011; opt.inc_y = 0b011'011'011;
        opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
        obj_grid(70, 10, &opt, &obj_ptr);
    }

    IF_TEST(3)
    {
        // No inc y flags
        opt.seq = 0b000'000'000;
        opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
        opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
        obj_grid(130, 10, &opt, &obj_ptr);
    }
        
    IF_TEST(4)
    {
        // Latch x always flags
        opt.seq = 0b000'000'000;
        opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
        opt.latch_x = 0b011'111'111; opt.inc_x = 0b000'100'100;   
        obj_grid(190, 10, &opt, &obj_ptr);
    }

    obj_master_xy(o, 140, 130); obj_commit_reset(o, &obj_ptr);

    // Now with zoom
    opt.zoom_x = 126; opt.zoom_y = 126;

    IF_TEST(5)
    {
        // "Standard" way
        opt.seq = 0b111'111'110;
        opt.latch_y = 0b011'011'011; opt.inc_y = 0b011'011'011;
        opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
        obj_grid(10, 10, &opt, &obj_ptr);
    }

    IF_TEST(6)
    {
        // No sequence flags
        opt.seq = 0b000'000'000;
        opt.latch_y = 0b011'011'011; opt.inc_y = 0b011'011'011;
        opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
        obj_grid(70, 10, &opt, &obj_ptr);
    }

    IF_TEST(7)
    {
        // No inc y flags
        opt.seq = 0b000'000'000;
        opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
        opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
        obj_grid(130, 10, &opt, &obj_ptr);
    }

    IF_TEST(8)
    {
        // Latch x always flags
        opt.seq = 0b000'000'000;
        opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
        opt.latch_x = 0b011'111'111; opt.inc_x = 0b000'100'100;   
        obj_grid(190, 10, &opt, &obj_ptr);
    }
            
    obj_master_xy(o, 140, 190); obj_commit_reset(o, &obj_ptr);
    
    IF_TEST(9)
    {
        // Now with zoom and always set seq
        // No inc y flags
        opt.seq = 0b111'111'110;
        opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
        opt.latch_x = 0b000'111'111; opt.inc_x = 0b000'100'100;   
        obj_grid(130, 10, &opt, &obj_ptr);
    }

    IF_TEST(10)
    {
        // Latch x always flags
        opt.seq = 0b111'111'110;
        opt.latch_y = 0b011'011'011; opt.inc_y = 0b000'000'000;
        opt.latch_x = 0b011'111'111; opt.inc_x = 0b000'100'100;   
        obj_grid(190, 10, &opt, &obj_ptr);
    }
    
    obj_cmd(o, cmd_base | OBJCMD_DISABLE); obj_commit_reset(o, &obj_ptr);
}

void init_basic_timing()
{
    reset_screen();
}

void update_basic_timing()
{
    uint32_t increment = 0;
    uint32_t scn_write = 0;
    uint32_t obj_write = 0;
    uint32_t dar_write = 0;
    uint32_t roz_write = 0;

    volatile uint16_t *scn_addr = &TC0100SCN->bg0[0].code;
    volatile uint16_t *obj_addr = &TC0200OBJ[64].pos0;
    volatile uint8_t *dar_addr = (uint8_t *)&TC0260DAR[128];
    volatile uint16_t *roz_addr = &TC0430GRW[128];

    uint32_t vb = vblank_count;
    while( vb == vblank_count ) {}
    vb = vblank_count;
    
    while( vb == vblank_count )
    {
        increment++;
    }
    vb = vblank_count;

    while( vb == vblank_count )
    {
        *scn_addr = 0x0000;
        scn_write++;
    }
    vb = vblank_count;

    while( vb == vblank_count )
    {
        *obj_addr = 0x0000;
        obj_write++;
    }
    vb = vblank_count;

    while( vb == vblank_count )
    {
        *dar_addr = 0x0000;
        dar_write++;
    }
    vb = vblank_count;

    while( vb == vblank_count )
    {
        *roz_addr = 0x0000;
        roz_write++;
    }
    vb = vblank_count;

    pen_color(0);
    on_layer(FG0);
    move_to(4, 4);
    print("INCREMENT: %08X\n", increment);
    print("SCN_WRITE: %08X\n", scn_write);
    print("OBJ_WRITE: %08X\n", obj_write);
    print("DAR_WRITE: %08X\n", dar_write);
    print("ROZ_WRITE: %08X\n", roz_write);
}

void init_360pri()
{
    reset_screen();

    fg0_gfx(2);
    fg0_row(1, 1, 1, 1, 1, 1, 1, 1);
    fg0_row(1, 0, 0, 0, 0, 0, 0, 1);
    fg0_row(1, 0, 2, 2, 2, 2, 0, 1);
    fg0_row(1, 0, 2, 3, 3, 2, 0, 1);
    fg0_row(1, 0, 2, 3, 3, 2, 0, 1);
    fg0_row(1, 0, 2, 2, 2, 2, 0, 1);
    fg0_row(1, 0, 0, 0, 0, 0, 0, 1);
    fg0_row(1, 1, 1, 1, 1, 1, 1, 1);

    fg0_row(1, 1, 1, 1, 1, 1, 1, 1);
    fg0_row(1, 1, 1, 1, 1, 1, 1, 1);
    fg0_row(1, 1, 1, 1, 1, 1, 1, 1);
    fg0_row(2, 2, 2, 2, 2, 2, 2, 2);
    fg0_row(2, 2, 2, 2, 2, 2, 2, 2);
    fg0_row(3, 3, 3, 3, 3, 3, 3, 3);
    fg0_row(3, 3, 3, 3, 3, 3, 3, 3);
    fg0_row(3, 3, 3, 3, 3, 3, 3, 3);

    fg0_row(1, 1, 2, 2, 3, 3, 1, 1);
    fg0_row(1, 1, 2, 2, 3, 3, 1, 1);
    fg0_row(1, 1, 2, 2, 3, 3, 1, 1);
    fg0_row(1, 1, 2, 2, 3, 3, 1, 1);
    fg0_row(1, 1, 2, 2, 3, 3, 1, 1);
    fg0_row(1, 1, 2, 2, 3, 3, 1, 1);
    fg0_row(1, 1, 2, 2, 3, 3, 1, 1);
    fg0_row(1, 1, 2, 2, 3, 3, 1, 1);

    fg0_row(1, 1, 1, 1, 3, 3, 3, 3);
    fg0_row(1, 1, 1, 1, 3, 3, 3, 3);
    fg0_row(1, 1, 1, 1, 3, 3, 3, 3);
    fg0_row(1, 1, 1, 1, 3, 3, 3, 3);
    fg0_row(2, 2, 2, 2, 0, 0, 0, 0);
    fg0_row(2, 2, 2, 2, 0, 0, 0, 0);
    fg0_row(2, 2, 2, 2, 0, 0, 0, 0);
    fg0_row(2, 2, 2, 2, 0, 0, 0, 0);

    tc0360pri_set_roz_prio(4, 4, 4, 4);
    tc0360pri_set_roz(0, 4 << 2);
    
    TC0200OBJ_Inst *obj_ptr = TC0200OBJ;
    TC0200OBJ_Inst work;
    TC0200OBJ_Inst *o = &work;
    uint16_t cmd_base = 0;
    obj_reset(o);
    obj_cmd(o, cmd_base); obj_commit_reset(o, &obj_ptr);
    obj_master_xy(o, 128, 23); obj_commit_reset(o, &obj_ptr);
 
    GridOptions opt;
    memset(&opt, 0, sizeof(opt));
    opt.w = 1; opt.h = 1;
    opt.extra = opt.zoom = 0b1;
    opt.tile = 0x7AE;
    
    for( int y = 0; y < 4; y++ )
    {
        for (int x = 0; x < 4; x++ )
        {
            opt.color = 24 + y + (64 * x);
            obj_grid(x * 24, y * 24, &opt, &obj_ptr);
            on_layer(FG0);
            pen_color(28);
            sym_at(5 + (3 * x), 2 + (3 * y), 3);
            sym_at(18 + (3 * x), 2 + (3 * y), 3);
            pen_color(29);
            sym_at(6 + (3 * x), 2 + (3 * y), 3);
            sym_at(19 + (3 * x), 2 + (3 * y), 3);
            pen_color(30);
            sym_at(5 + (3 * x), 3 + (3 * y), 3);
            sym_at(18 + (3 * x), 3 + (3 * y), 3);
            pen_color(31);
            sym_at(6 + (3 * x), 3 + (3 * y), 3);
            sym_at(19 + (3 * x), 3 + (3 * y), 3);
             
            on_layer(ROZ);
            pen_color(3);
            sym_at(19 + (3 * x), 2 + (3 * y), 0x67);
            sym_at(18 + (3 * x), 3 + (3 * y), 0x67);
        }
    }

    opt.color = 16;
    obj_grid(10 * 24, 0 * 24, &opt, &obj_ptr);

    on_layer(FG0);

    for( int y = 0; y < 8; y++ )
    {
        pen_color(24 + y);
        sym_at(31, y+2, 0x1);
    }
}

#define WRAP(v, mx) v = (v >= 255 || v < 0) ? (mx) : (v > (mx)) ? (0) : v

void update_360pri()
{
    static int sel = 0;

    static uint8_t fg_prio = 6;
    static uint8_t obj0_prio = 2;
    static uint8_t obj1_prio = 3;
    static uint8_t obj2_prio = 5;
    static uint8_t obj3_prio = 7;
    static uint8_t roz_prio = 4;
    static uint8_t black_out = 0;
    static uint8_t white_out = 0;

    static uint8_t blend_mode = 0;

    wait_vblank();

    on_layer(FG0);
    pen_color(8);
    move_to(4, 15);

    print("%c FG: %X\n", sel == 0 ? '*' : ' ', fg_prio);
    print("%c O1: %X\n", sel == 1 ? '*' : ' ', obj0_prio);
    print("%c O2: %X\n", sel == 2 ? '*' : ' ', obj1_prio);
    print("%c O3: %X\n", sel == 3 ? '*' : ' ', obj2_prio);
    print("%c O4: %X\n", sel == 4 ? '*' : ' ', obj3_prio);
    print("%c RZ: %X\n", sel == 5 ? '*' : ' ', roz_prio);

    move_to(15, 15);
    for( int x = 0; x < 8; x++ )
    {
        print("%c BM%d: %X\n", sel == (6 + x) ? '*' : ' ', x, (blend_mode >> x) & 0x1);
    }
    
    move_to(24,15);
    print("%c BLACK: %02X\n", sel == 14 ? '*' : ' ', black_out);
    print("%c WHITE: %02X\n", sel == 15 ? '*' : ' ', white_out);


    int adj = 0;
    bool color_change = false;
    if (input_pressed(LEFT)) adj = -1;
    if (input_pressed(RIGHT)) adj = 1;

    switch(sel)
    {
        case 0: fg_prio += adj; break;
        case 1: obj0_prio += adj; break;
        case 2: obj1_prio += adj; break;
        case 3: obj2_prio += adj; break;
        case 4: obj3_prio += adj; break;
        case 5: roz_prio += adj; break;
        case 14: black_out += adj; color_change = adj != 0; break;
        case 15: white_out += adj; color_change = adj != 0; break;
        default: break;
    }

    if (adj != 0 && sel >= 6 && sel <= 13)
    {
        blend_mode ^= (1 << (sel - 6));
    }

    if (input_pressed(DOWN)) sel++;
    if (input_pressed(UP)) sel--;

    WRAP(sel, 15);
    WRAP(fg_prio, 15);
    WRAP(obj0_prio, 15);
    WRAP(obj1_prio, 15);
    WRAP(obj2_prio, 15);
    WRAP(obj3_prio, 15);
    WRAP(roz_prio, 15);

    if (color_change)
    {
        set_default_palette();
        set_gradient2(white_out, 255, 255, 255, 255, 255, 255);
        set_gradient2(black_out, 0, 0, 0, 0, 0, 0);
    }

    tc0360pri_set_roz_prio(roz_prio, roz_prio, roz_prio, roz_prio);
    tc0360pri_set_obj_prio(obj0_prio, obj1_prio, obj2_prio, obj3_prio);
    tc0360pri_set_tile_prio(fg_prio, 0, 0);
    tc0360pri_set_blendmode(blend_mode);
}

void init_screen(int screen)
{
    switch(screen)
    {
        case 0: init_scn_general(); break;
        case 1: init_scn_control_access(); break;
        case 2: init_obj_general(); break;
        case 3: init_obj_test1(); break;
        case 4: init_obj_test2(); break;
        case 5: init_obj_test3(); break;
        case 6: init_sound_test(); break;
        case 7: init_scn_align(); break;
        case 8: init_basic_timing(); break;
        case 9: init_360pri(); break;
        default: break;
    }
}

void update_screen(int screen)
{
    switch(screen)
    {
        case 0: update_scn_general(); break;
        case 1: update_scn_control_access(); break;
        case 2: update_obj_general(); break;
        case 3: update_obj_test1(); break;
        case 4: update_obj_test2(); break;
        case 5: update_obj_test3(); break;
        case 6: update_sound_test(); break;
        case 7: update_scn_align(); break;
        case 8: update_basic_timing(); break;
        case 9: update_360pri(); break;
        default: break;
    }
}

void deinit_screen(int screen)
{
    switch(screen)
    {
        default: break;
    }
}

int main(int argc, char *argv[])
{
    enable_interrupts();

    uint32_t system_flags = 0;

    int current_screen = 9;

    init_screen(current_screen);
    
    while(1)
    {
        input_update();

        // 0x0200 // ACC MODE
        // 0x0100 // Brightness? 
        // 0x0002 // Z4
        // 0x0001 // Z3
        //
        //*(uint16_t*)0x500000 = 0x0000;

        if (input_pressed(START))
        {
            deinit_screen(current_screen);
            current_screen = ( current_screen + 1 ) % NUM_SCREENS;
            init_screen(current_screen);
        }

        update_screen(current_screen);
    }

    return 0;
}


