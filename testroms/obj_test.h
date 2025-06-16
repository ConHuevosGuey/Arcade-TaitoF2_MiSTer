#ifndef OBJ_TEST_H
#define OBJ_TEST_H 1

#include <stdint.h>

#define GRD_ALWAYS  0xffff
#define GRD_NEVER   0x0000

typedef struct
{
    int16_t  w;
    int16_t  h;
    uint8_t  zoom_x;
    uint8_t  zoom_y;
    uint16_t seq;
    uint16_t inc_x;
    uint16_t inc_y;
    uint16_t latch_x;
    uint16_t latch_y;
    uint16_t zoom;
    uint16_t extra;
    uint16_t pos;
    uint16_t tile;
    uint16_t color;
} GridOptions;

void obj_grid(int x, int y, const GridOptions *opts, TC0200OBJ_Inst **ptr);

#endif
