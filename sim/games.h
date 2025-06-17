#ifndef GAMES_H
#define GAMES_H 1

#include <stdint.h>

enum game_t : uint8_t
{
    GAME_FINALB = 0,
    GAME_DONDOKOD,
    GAME_MEGAB,
    GAME_THUNDFOX,
    GAME_CAMELTRY,
    GAME_QTORIMON,
    GAME_LIQUIDK,
    GAME_QUIZHQ,
    GAME_SSI,
    GAME_GUNFRONT,
    GAME_GROWL,
    GAME_MJNQUEST,
    GAME_FOOTCHMP,
    GAME_KOSHIEN,
    GAME_YUYUGOGO,
    GAME_NINJAK,
    GAME_SOLFIGTR,
    GAME_QZQUEST,
    GAME_PULIRULA,
    GAME_METALB,
    GAME_QZCHIKYU,
    GAME_YESNOJ,
    GAME_DEADCONX,
    GAME_DINOREX,
    GAME_QJINSEI,
    GAME_QCRAYON,
    GAME_QCRAYON2,
    GAME_DRIFTOUT,

    GAME_FINALB_TEST,
    GAME_QJINSEI_TEST,
    GAME_DRIFTOUT_TEST,

    N_GAMES,

    GAME_INVALID = 0xff
};

static const uint32_t CPU_ROM_SDR_BASE      = 0x00000000;
static const uint32_t SCN0_ROM_SDR_BASE     = 0x00900000;
static const uint32_t SCN1_ROM_SDR_BASE     = 0x01200000;
static const uint32_t ADPCMA_ROM_SDR_BASE   = 0x00b00000;
static const uint32_t ADPCMB_ROM_SDR_BASE   = 0x00d00000;
static const uint32_t PIVOT_ROM_SDR_BASE    = 0x01000000;
static const uint32_t OBJ_DATA_DDR_BASE     = 0x00200000;

game_t game_find(const char *name);
const char *game_name(game_t game);

bool game_init(game_t game);


#endif // GAMES_H
