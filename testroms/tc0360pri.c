#include "tc0360pri.h"

#include "system.h"

static uint8_t blend_mode = 0;

void tc0360pri_vblank()
{
    TC0360PRI->blend_mode = blend_mode;
}

void tc0360pri_set_blendmode(uint8_t bm)
{
    blend_mode = bm;
}

void tc0360pri_set_roz(uint8_t priority, uint8_t color)
{
    TC0360PRI->roz_prio_color = (priority << 6) | (color >> 2);
}

void tc0360pri_set_tile_prio(uint8_t fg0, uint8_t bg0, uint8_t bg1)
{
    TC0360PRI->prio_bg1_bg0 = (bg1 << 4) | (bg0 & 0x0f);
    TC0360PRI->prio_fg1_fg0 = (fg0 << 4) | 0;
}

void tc0360pri_set_obj_prio(uint8_t p00, uint8_t p01, uint8_t p10, uint8_t p11)
{
    TC0360PRI->prio_obj_01_00 = (p01 << 4) | (p00 & 0x0f);
    TC0360PRI->prio_obj_11_10 = (p11 << 4) | (p10 & 0x0f);
}

void tc0360pri_set_roz_prio(uint8_t p00, uint8_t p01, uint8_t p10, uint8_t p11)
{
    TC0360PRI->prio_roz_01_00 = (p01 << 4) | (p00 & 0x0f);
    TC0360PRI->prio_roz_11_10 = (p11 << 4) | (p10 & 0x0f);
}

