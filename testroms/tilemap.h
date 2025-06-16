#if !defined(TILEMAP_H)
#define TILEMAP_H 1

#include <stdint.h>

typedef enum
{
    BG0 = 0,
    BG1,
    FG0,
    ROZ,
} Layer;

void on_layer(Layer layer);
void pen_color(int color);
void move_to(int x, int y);
void print(const char *fmt, ...);
void print_at(int x, int y, const char *fmt, ...);
void sym_at(int x, int y, uint16_t sym);

void fg0_gfx(int ch);
void fg0_row(uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3, uint8_t x4, uint8_t x5, uint8_t x6, uint8_t x7);

#endif
