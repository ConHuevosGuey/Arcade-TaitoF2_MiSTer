#include <stdint.h>
#include <stdarg.h>

#include "printf/printf.h"
#include "tilemap.h"
#include "system.h"

#define SCN TC0100SCN

Layer cur_layer;
uint16_t cur_x, cur_y;
uint16_t cur_color;

void on_layer(Layer layer)
{
    cur_layer = layer;
}

void pen_color(int color)
{
    cur_color = color;
}

void move_to(int x, int y)
{
    cur_x = x;
    cur_y = y;
}

static void print_string(const char *str)
{
    uint16_t x = cur_x;
    uint16_t y = cur_y;

    uint16_t ofs = ( y * 64 ) + x;

    uint16_t attr_color = (0 << 8) | (cur_color & 0xff);

    while(*str)
    {
        if( *str == '\n' )
        {
            y++;
            x = cur_x;
            ofs = (y * 64) + x;
        }
        else
        {
            switch(cur_layer)
            {
                case BG0:
                    SCN->bg0[ofs].attr_color = attr_color;
                    SCN->bg0[ofs].code = *str;
                    break;

                case BG1:
                    SCN->bg1[ofs].attr_color = attr_color;
                    SCN->bg1[ofs].code = *str;
                    break;

                case FG0:
                    SCN->fg0[ofs].attr = cur_color & 0x3f;
                    SCN->fg0[ofs].code = *str;
                    break;

                case ROZ:
                    TC0430GRW[ofs] = (cur_color << 14) | *str;
                    break;
                default:
                    break;
            }
            ofs++;
            x++;
        }
        str++;
    }
    cur_x = x;
    cur_y = y;
}

void print(const char *fmt, ...)
{
    char buf[128];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    buf[127] = '\0';
    va_end(args);
    
    print_string(buf);
}

void print_at(int x, int y, const char *fmt, ...)
{
    char buf[128];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    buf[127] = '\0';
    va_end(args);
    
    move_to(x, y);
    print_string(buf);
}

void sym_at(int x, int y, uint16_t sym)
{
    uint16_t ofs = ( y * 64 ) + x;
    uint16_t attr_color = ( 0 << 8 ) | ( cur_color & 0xff );
    
    switch(cur_layer)
    {
        case BG0:
            SCN->bg0[ofs].attr_color = attr_color;
            SCN->bg0[ofs].code = sym;
            break;

        case BG1:
            SCN->bg1[ofs].attr_color = attr_color;
            SCN->bg1[ofs].code = sym;
            break;

        case FG0:
            SCN->fg0[ofs].attr = cur_color & 0x3f;
            SCN->fg0[ofs].code = sym;
            break;
        
        case ROZ:
            TC0430GRW[ofs] = (cur_color << 14) | sym;
            break;

        default:
            break;
    }
}

uint16_t *fg0_ptr = NULL;

void fg0_gfx(int ch)
{
    fg0_ptr = SCN->fg0_gfx + (8 * ch);
}

void fg0_row(uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3, uint8_t x4, uint8_t x5, uint8_t x6, uint8_t x7)
{
    uint16_t gfx;
    gfx  = (x0&1) << 7 | (x1&1) << 6 | (x2&1) << 5 | (x3&1) << 4 | (x4&1) << 3 | (x5&1) << 2 | (x6&1) << 1 | (x7&1) << 0;
    gfx |= (x0&2) << 14 | (x1&2) << 13 | (x2&2) << 12 | (x3&2) << 11 | (x4&2) << 10 | (x5&2) << 9 | (x6&2) << 8 | (x7&2) << 7;
    *fg0_ptr = gfx;
    fg0_ptr++;
}

