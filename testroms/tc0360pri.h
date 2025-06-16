#if !defined( TC0360PRI_H )
#define TC0360PRI_H 1

#include <stdint.h>

typedef struct TC0360PRI_Control
{
    uint8_t blend_mode;
    uint8_t _pad0;

    uint8_t roz_prio_color;
    uint8_t _pad1;

    uint8_t unk1;
    uint8_t _pad2;

    uint8_t unk2;
    uint8_t _pad3;

    uint8_t prio_fg1_fg0;
    uint8_t _pad4;

    uint8_t prio_bg1_bg0;
    uint8_t _pad5;

    uint8_t prio_obj_01_00;
    uint8_t _pad6;
    
    uint8_t prio_obj_11_10;
    uint8_t _pad7;

    uint8_t prio_roz_01_00;
    uint8_t _pad8;
    
    uint8_t prio_roz_11_10;
    uint8_t _pad9;

    uint8_t unk3;
    uint8_t _pada;

    uint8_t unk4;
    uint8_t _padb;

    uint8_t unk5;
    uint8_t _padc;

    uint8_t unk6;
    uint8_t _padd;

    uint8_t unk7;
    uint8_t _pade;

    uint8_t unk8;
    uint8_t _padf;

} TC0360PRI_Control;

_Static_assert(sizeof(TC0360PRI_Control) == 0x20, "TC0360PRI_Control mismatch");

void tc0360pri_vblank();
void tc0360pri_set_blendmode(uint8_t bm);
void tc0360pri_set_roz(uint8_t priority, uint8_t color);
void tc0360pri_set_tile_prio(uint8_t fg0, uint8_t bg0, uint8_t bg1);
void tc0360pri_set_obj_prio(uint8_t p00, uint8_t p01, uint8_t p10, uint8_t p11);
void tc0360pri_set_roz_prio(uint8_t p00, uint8_t p01, uint8_t p10, uint8_t p11);


#endif
