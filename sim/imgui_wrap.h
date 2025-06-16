#if !defined(IMGUI_WRAP_H)
#define IMGUI_WRAP_H 1

#define IMGUI_DEFINE_MATH_OPERATORS

#include "imgui.h"
#include "imgui_internal.h"

bool imgui_init(const char *title);
bool imgui_begin_frame();
void imgui_end_frame();

uint32_t imgui_get_buttons();

struct SDL_Renderer;
SDL_Renderer *imgui_get_renderer();

#endif // IMGUI_WRAP_H
