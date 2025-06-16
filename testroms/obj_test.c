#include "tc0200obj.h"
#include "obj_test.h"
#include "system.h"

void obj_grid(int x, int y, const GridOptions *opts, TC0200OBJ_Inst **ptr)
{
    int count = opts->w * opts->h;
    int idx = 0;
    uint16_t bit;
    uint16_t tile = opts->tile ? opts->tile : 0x1a4e;

    bit = 1 << (count-1);

    TC0200OBJ_Inst work;
    TC0200OBJ_Inst *o = &work;

    for( int xx = 0; xx < opts->w; xx++ )
    {
        for( int yy = 0; yy < opts->h; yy++ )
        {
            obj_reset(o);
            if ( opts->extra & bit ) obj_extra_xy(o, x, y);
            if ( opts->pos & bit ) obj_xy(o, x, y);
            if ( opts->zoom & bit )
            {
                o->zoom_x = opts->zoom_x;
                o->zoom_y = opts->zoom_y;
            }

            o->code = tile + (idx % 10);
            o->color = opts->color;
            o->is_seq  = opts->seq & bit ? 1 : 0;
            o->inc_x   = opts->inc_x & bit ? 1 : 0;
            o->inc_y   = opts->inc_y & bit ? 1 : 0;
            o->latch_x = opts->latch_x & bit ? 1 : 0;
            o->latch_y = opts->latch_y & bit ? 1 : 0;

            obj_commit(o, ptr);
            idx++;
            bit >>= 1;
        }
    }
}


