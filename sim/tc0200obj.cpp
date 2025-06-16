#include <SDL.h>

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_wrap.h"
#include "tc0200obj.h"

#include "F2.h"
#include "F2___024root.h"

extern F2* top;

static SDL_Renderer *s_renderer = nullptr;
static uint64_t s_used_idx = 0;
static const uint8_t *s_palette_low = nullptr;
static const uint8_t *s_palette_high = nullptr;
static const uint8_t *s_objmem = nullptr;


struct ObjCacheEntry
{
    SDL_Texture *texture;
    uint64_t last_used;

    ObjCacheEntry() : texture(nullptr), last_used(0) {}

    ObjCacheEntry(SDL_Texture *tex)
        : texture(tex)
        , last_used(0) {}

    ObjCacheEntry(ObjCacheEntry&& o)
    {
        texture = o.texture;
        last_used = o.last_used;
        o.texture = nullptr;
    }

    ~ObjCacheEntry()
    {
        if (texture)
            SDL_DestroyTexture(texture);
    }

    ObjCacheEntry(const ObjCacheEntry&) = delete;

    SDL_Texture *deref()
    {
        last_used = s_used_idx;
        s_used_idx++;
        return texture;
    }
};

static std::map<uint64_t, ObjCacheEntry> obj_cache;

#define FNV_64_PRIME ((uint64_t)0x100000001b3ULL)

uint64_t fnv_hash(void *buf, size_t len, uint64_t hval)
{
    unsigned char *bp = (unsigned char *)buf;
    unsigned char *be = bp + len;

    while (bp < be) {
	    hval *= FNV_64_PRIME;
	    hval ^= (uint64_t)*bp++;
    }

    return hval;
}

void init_obj_cache(SDL_Renderer *renderer, const void *objmem, const void *palmem_low, const void *palmem_high)
{
    s_renderer = renderer;
    s_objmem = (const uint8_t *)objmem;
    s_palette_low = (const uint8_t *)palmem_low;
    s_palette_high = (const uint8_t *)palmem_high;
}

void prune_obj_cache()
{
    if (obj_cache.size() < 2048) return;

    size_t num_to_remove = 128;

    std::vector<std::pair<uint64_t, uint64_t>> hash_ages;
    for( const auto& it : obj_cache )
    {
        hash_ages.push_back({it.second.last_used, it.first});
    }
    std::sort(hash_ages.begin(), hash_ages.end());
    for( size_t i = 0; i < num_to_remove; i++ )
    {
        obj_cache.erase(hash_ages[i].second);
    }
}

SDL_Texture *get_obj_texture(uint16_t code, uint8_t palette)
{
    uint16_t pal_ofs = palette * 16;

    uint16_t rawpal[16];

    for( int i = 0; i < 16; i++ )
    {
        rawpal[i] = (s_palette_high[pal_ofs + i] << 8) | (s_palette_low[pal_ofs + i] << 0);
    }

    uint64_t hash = fnv_hash(rawpal, sizeof(rawpal), code);
    auto it = obj_cache.find(hash);
    if (it != obj_cache.end())
    {
        return it->second.deref();
    }

    uint32_t pal32[16];
    for( int i = 0; i < 16; i++ )
    {
#if 0
        uint8_t r = ((rawpal[i] & 0xf000) >> 8) | ((rawpal[i] & 0x0008) >> 0);
        uint8_t g = ((rawpal[i] & 0x0f00) >> 4) | ((rawpal[i] & 0x0004) << 1);
        uint8_t b = ((rawpal[i] & 0x00f0) >> 0) | ((rawpal[i] & 0x0002) << 2);
#else
        uint8_t r = ((rawpal[i] & 0x7c00) >> 7);
        uint8_t g = ((rawpal[i] & 0x03e0) >> 2);
        uint8_t b = ((rawpal[i] & 0x001f) << 3);
#endif

        uint32_t c = (r << 24) | (g << 16) | (b << 8);
        pal32[i] = c;
    }

    SDL_Texture *tex = SDL_CreateTexture(s_renderer, SDL_PIXELFORMAT_RGBX8888, SDL_TEXTUREACCESS_STATIC, 16, 16);

    const uint8_t *src = s_objmem + (code * 128);
    uint32_t pixels[16 * 16];
    uint32_t *dest = pixels;

    for( int i = 0; i < 128; i++ )
    {
        dest[1] = pal32[((*src & 0xf0) >> 4)];
        dest[0] = pal32[((*src & 0x0f) >> 0)];
        dest += 2;
        src++;
    }

    SDL_UpdateTexture(tex, nullptr, pixels, 16 * 4);

    obj_cache.emplace(hash, ObjCacheEntry(tex));

    return obj_cache[hash].deref();
}


void get_obj_inst(uint16_t index, TC0200OBJ_Inst *inst)
{
    uint8_t *inst_data = (uint8_t *)inst;

    uint16_t offset = index * 8;

    for( int i = 0; i < 8; i++ )
    {
        inst_data[(i * 2) + 0] = top->rootp->F2__DOT__obj_ram__DOT__ram_l.m_storage[offset + i];
        inst_data[(i * 2) + 1] = top->rootp->F2__DOT__obj_ram__DOT__ram_h.m_storage[offset + i];
    }
}

uint16_t extended_code(uint16_t index, uint16_t code)
{
    if (top->rootp->F2__DOT__cfg_obj_extender == 1)
    {
        uint8_t ext = top->rootp->F2__DOT__tc0200obj_extender__DOT__extension_ram__DOT__ram.m_storage[index];
        return (code & 0xff) | (ext << 8);
    }
    else
    {
        return code;
    }
}

static void bullet(int x)
{
    if (x != 0) ImGui::Bullet();
}

void draw_obj_window()
{
    static int bank = 0;
    const char *bank_names[4] = { "0x0000", "0x4000", "0x8000", "0xC000" };

    if( !ImGui::Begin("TC0200OBJ") )
    {
        ImGui::End();
        return;
    }

    ImGui::Combo("Bank", &bank, bank_names, 4);

    if( ImGui::BeginTable("obj", 26, ImGuiTableFlags_HighlightHoveredColumn |
                      ImGuiTableFlags_BordersInnerV |
                      ImGuiTableFlags_ScrollY |
                      ImGuiTableFlags_SizingFixedFit |
                      ImGuiTableFlags_RowBg) )
    {
        uint32_t colflags = ImGuiTableColumnFlags_AngledHeader;
        ImGui::TableSetupColumn("", colflags & ~ImGuiTableColumnFlags_AngledHeader);
        ImGui::TableSetupColumn("Code", colflags);
        ImGui::TableSetupColumn("Code (Ext)", colflags);
        ImGui::TableSetupColumn("X", colflags);
        ImGui::TableSetupColumn("Y", colflags);
        ImGui::TableSetupColumn("Cmd", colflags);
        ImGui::TableSetupColumn("Latch Extra", colflags);
        ImGui::TableSetupColumn("Latch Master", colflags);
        ImGui::TableSetupColumn("Ignore Extra", colflags);
        ImGui::TableSetupColumn("Ignore All", colflags);
        ImGui::TableSetupColumn("Flip X", colflags);
        ImGui::TableSetupColumn("Flip Y", colflags);
        ImGui::TableSetupColumn("Is Seq", colflags);
        ImGui::TableSetupColumn("Latch Y", colflags);
        ImGui::TableSetupColumn("Inc Y", colflags);
        ImGui::TableSetupColumn("Latch X", colflags);
        ImGui::TableSetupColumn("Inc X", colflags);
        ImGui::TableSetupColumn("Latch Color", colflags);
        ImGui::TableSetupColumn("Color", colflags);
        ImGui::TableSetupColumn("Unk1", colflags);
        ImGui::TableSetupColumn("Unk2", colflags);
        ImGui::TableSetupColumn("Unk3", colflags);
        ImGui::TableSetupColumn("Cmd", colflags);
        ImGui::TableSetupColumn("Zoom X", colflags);
        ImGui::TableSetupColumn("Zoom Y", colflags);
        ImGui::TableSetupColumn("Debug", colflags);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableAngledHeadersRow();

        TC0200OBJ_Inst insts[1024];
        uint8_t latched_color[1024];
        uint16_t extcode[1024];

        const int bank_base = bank * 1024;
        
        uint8_t last_color = 0;
        for (int i = 0; i < 1024; i++)
        {
            get_obj_inst(bank_base + i, &insts[i]);
            extcode[i] = extended_code(bank_base + i, insts[i].code);
            if (insts[i].latch_color)
            {
                latched_color[i] = last_color;
            }
            else
            {
                latched_color[i] = last_color = insts[i].color;
            }
        }

        int hovered = ImGui::TableGetHoveredRow();

        ImGuiListClipper clipper;
        clipper.Begin(1024);
        int row_count = 1;
        int tooltip_idx = -1;
        while( clipper.Step() )
        {
            for( uint16_t index = clipper.DisplayStart; index < clipper.DisplayEnd; index++ )
            {
                if (row_count == hovered)
                {
                    tooltip_idx = index;
                }
                row_count++;

                TC0200OBJ_Inst& inst = insts[index];
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::Text("%4u", index);
                ImGui::TableNextColumn(); ImGui::Text("%04X", inst.code);
                ImGui::TableNextColumn(); ImGui::Text("%04X", extcode[index]);
                ImGui::TableNextColumn(); ImGui::Text("%4d", inst.x);
                ImGui::TableNextColumn(); ImGui::Text("%4d", inst.y);
                ImGui::TableNextColumn(); bullet(inst.has_cmd);
                ImGui::TableNextColumn(); bullet(inst.latch_extra);
                ImGui::TableNextColumn(); bullet(inst.latch_master);
                ImGui::TableNextColumn(); bullet(inst.ignore_extra);
                ImGui::TableNextColumn(); bullet(inst.ignore_all);
                ImGui::TableNextColumn(); bullet(inst.flipx);
                ImGui::TableNextColumn(); bullet(inst.flipy);
                ImGui::TableNextColumn(); bullet(inst.is_seq);
                ImGui::TableNextColumn(); bullet(inst.latch_y);
                ImGui::TableNextColumn(); bullet(inst.inc_y);
                ImGui::TableNextColumn(); bullet(inst.latch_x);
                ImGui::TableNextColumn(); bullet(inst.inc_x);
                ImGui::TableNextColumn(); bullet(inst.latch_color);
                ImGui::TableNextColumn(); ImGui::Text("%02X", inst.color);
                ImGui::TableNextColumn(); bullet(inst.unk1);
                ImGui::TableNextColumn(); bullet(inst.unk2);
                ImGui::TableNextColumn(); bullet(inst.unk3);
                ImGui::TableNextColumn(); ImGui::Text("%04X", inst.cmd_bits);
                ImGui::TableNextColumn(); ImGui::Text("%02X", inst.zoom_x);
                ImGui::TableNextColumn(); ImGui::Text("%02X", inst.zoom_y);
                ImGui::TableNextColumn();
                bool is_debug = index == top->obj_debug_idx;
                char id[16];
                snprintf(id, 16, "##debug%d", index);
                if (ImGui::RadioButton(id, is_debug))
                {
                    if (is_debug)
                        top->obj_debug_idx = -1;
                    else
                        top->obj_debug_idx = index;
                }

            }
        }

        if (tooltip_idx >= 0)
        {
            uint16_t code = extcode[tooltip_idx];
            if (code != 0)
            {
                SDL_Texture *tex = get_obj_texture(code, latched_color[tooltip_idx]);
                ImGui::BeginTooltip();
                ImGui::LabelText("Code", "%04X", code);
                ImGui::LabelText("Color", "%02X", latched_color[tooltip_idx]);
                ImGui::Image((ImTextureID)tex, ImVec2(64, 64));
                ImGui::End();
            }
        }


        ImGui::EndTable();
    }

    ImGui::End();
}

void draw_obj_preview_window()
{
    static int color = 0;

    if( !ImGui::Begin("TC0200OBJ Preview") )
    {
        ImGui::End();
        return;
    }
    
    ImGui::SliderInt("Color", &color, 0, 0xff, "%02X");
    color &= 0xff;

    if( ImGui::BeginTable("obj_list", 17, ImGuiTableFlags_HighlightHoveredColumn |
                      ImGuiTableFlags_BordersInnerV |
                      ImGuiTableFlags_ScrollY |
                      ImGuiTableFlags_SizingFixedFit))
    {
        ImGuiTableColumnFlags colflags = 0;
        ImGui::TableSetupColumn("", colflags);
        ImGui::TableSetupColumn("0", colflags);
        ImGui::TableSetupColumn("1", colflags);
        ImGui::TableSetupColumn("2", colflags);
        ImGui::TableSetupColumn("3", colflags);
        ImGui::TableSetupColumn("4", colflags);
        ImGui::TableSetupColumn("5", colflags);
        ImGui::TableSetupColumn("6", colflags);
        ImGui::TableSetupColumn("7", colflags);
        ImGui::TableSetupColumn("8", colflags);
        ImGui::TableSetupColumn("9", colflags);
        ImGui::TableSetupColumn("A", colflags);
        ImGui::TableSetupColumn("B", colflags);
        ImGui::TableSetupColumn("C", colflags);
        ImGui::TableSetupColumn("D", colflags);
        ImGui::TableSetupColumn("E", colflags);
        ImGui::TableSetupColumn("F", colflags);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(0x10000 / 16);
        while( clipper.Step() )
        {
            for( uint16_t index = clipper.DisplayStart; index < clipper.DisplayEnd; index++ )
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::Text("%03Xx", index);
                uint16_t base_code = index * 16;
                for( int i = 0; i < 16; i++)
                {
                    ImGui::TableNextColumn(); 
                    SDL_Texture *tex = get_obj_texture((uint16_t)base_code + i, (uint8_t)color);
                    ImGui::Image((ImTextureID)tex, ImVec2(32, 32));
                }
            }
        }


        ImGui::EndTable();
    }

    ImGui::End();
}


