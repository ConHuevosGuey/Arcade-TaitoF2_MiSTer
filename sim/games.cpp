#include "games.h"
#include "sim_sdram.h"
#include "sim_ddr.h"

#include "F2.h"
#include "F2___024root.h"

#include "file_search.h"
#include <string.h>

extern F2 *top;

static const char *game_names[N_GAMES] =
{
    "finalb",
    "dondokod",
    "megab",
    "thundfox",
    "cameltry",
    "qtorimon",
    "liquidk",
    "quizhq",
    "ssi",
    "gunfront",
    "growl",
    "mjnquest",
    "footchmp",
    "koshien",
    "yuyugogo",
    "ninjak",
    "solfigtr",
    "qzquest",
    "pulirula",
    "metalb",
    "qzchikyu",
    "yesnoj",
    "deadconx",
    "dinorex",
    "qjinsei",
    "qcrayon",
    "qcrayon2",
    "driftout",
    "finalb_test",
    "qjinsei_test",
    "driftout_test",
};


game_t game_find(const char *name)
{
    for( int i = 0; i < N_GAMES; i++ )
    {
        if (!strcasecmp(name, game_names[i]))
        {
            return (game_t)i;
        }
    }

    return GAME_INVALID;
}

const char *game_name(game_t game)
{
    if (game == GAME_INVALID) return "INVALID";
    return game_names[game];
}

static bool load_audio(const char *name)
{
    std::vector<uint8_t> data;
    if (!g_fs.loadFile(name, data))
    {
        printf("Could not open audio rom %s\n", name);
        return false;
    }

    memcpy(top->rootp->F2__DOT__sound_rom__DOT__ram.m_storage, data.data(), data.size());

    return true;
}

static void load_finalb()
{
    g_fs.addSearchPath("../roms/finalb.zip");

    load_audio("b82_10.ic5");

    sdram.load_data("b82-09.ic23", CPU_ROM_SDR_BASE + 1, 2);
    sdram.load_data("b82-17.ic11", CPU_ROM_SDR_BASE + 0, 2);

    sdram.load_data("b82-07.ic34", SCN0_ROM_SDR_BASE + 1, 2);
    sdram.load_data("b82-06.ic33", SCN0_ROM_SDR_BASE + 0, 2);
    
    sdram.load_data("b82-02.ic1",  ADPCMA_ROM_SDR_BASE, 1);
    sdram.load_data("b82-01.ic2",  ADPCMB_ROM_SDR_BASE, 1);

    ddr_memory.load_data("b82-03.ic9", OBJ_DATA_DDR_BASE + 0, 4);
    ddr_memory.load_data("b82-04.ic8", OBJ_DATA_DDR_BASE + 1, 4);
    ddr_memory.load_data("b82-05.ic7", OBJ_DATA_DDR_BASE + 2, 4);
    
    top->game = GAME_FINALB;
}

static void load_finalb_test()
{
    g_fs.addSearchPath("../testroms/build/finalb_test/finalb/");
    load_finalb();
}


static void load_qjinsei()
{
    g_fs.addSearchPath("../roms/qjinsei.zip");

    load_audio("d48-11");

    sdram.load_data("d48-09", CPU_ROM_SDR_BASE + 1, 2);
    sdram.load_data("d48-10", CPU_ROM_SDR_BASE + 0, 2);
    sdram.load_data("d48-03", CPU_ROM_SDR_BASE + 0x100000, 1);

    sdram.load_data("d48-04", SCN0_ROM_SDR_BASE, 1);
    
    sdram.load_data("d48-05",  ADPCMA_ROM_SDR_BASE, 1);

    ddr_memory.load_data("d48-02", OBJ_DATA_DDR_BASE + 0, 2);
    ddr_memory.load_data("d48-01", OBJ_DATA_DDR_BASE + 1, 2);
    
    top->game = GAME_QJINSEI;
}

static void load_qjinsei_test()
{
    g_fs.addSearchPath("../testroms/build/qjinsei_test/qjinsei");

    load_qjinsei();
}


static void load_dinorex()
{
    g_fs.addSearchPath("../roms/dinorex.zip");

    load_audio("d39-12.5");

    sdram.load_data("d39-14.9", CPU_ROM_SDR_BASE + 1, 2);
    sdram.load_data("d39-16.8", CPU_ROM_SDR_BASE + 0, 2);
    sdram.load_data("d39-04.6", CPU_ROM_SDR_BASE + 0x100000, 1);
    sdram.load_data("d39-05.7", CPU_ROM_SDR_BASE + 0x200000, 1);

    sdram.load_data("d39-06.2", SCN0_ROM_SDR_BASE, 1);

    sdram.load_data("d39-07.10",  ADPCMA_ROM_SDR_BASE, 1);
    sdram.load_data("d39-08.4",  ADPCMB_ROM_SDR_BASE, 1);

    ddr_memory.load_data("d39-01.29", OBJ_DATA_DDR_BASE, 1);
    ddr_memory.load_data("d39-02.28", OBJ_DATA_DDR_BASE + 0x200000, 1);
    ddr_memory.load_data("d39-03.27", OBJ_DATA_DDR_BASE + 0x400000, 1);

    top->game = GAME_DINOREX;
}

static void load_liquidk()
{
    g_fs.addSearchPath("../roms/liquidk.zip");
    load_audio("c49-08.ic32");

    sdram.load_data("c49-09.ic47", CPU_ROM_SDR_BASE + 1, 2);
    sdram.load_data("c49-11.ic48", CPU_ROM_SDR_BASE + 0, 2);
    sdram.load_data("c49-10.ic45", CPU_ROM_SDR_BASE + 0x40001, 2);
    sdram.load_data("c49-12.ic46", CPU_ROM_SDR_BASE + 0x40000, 2);
	
    sdram.load_data("c49-03.ic76", SCN0_ROM_SDR_BASE, 1);
   
    sdram.load_data("c49-04.ic33",  ADPCMA_ROM_SDR_BASE, 1);

    ddr_memory.load_data("c49-01.ic54", OBJ_DATA_DDR_BASE, 1);
    ddr_memory.load_data("c49-02.ic53", OBJ_DATA_DDR_BASE + 0x80000, 1);

    top->game = GAME_LIQUIDK;
}

static void load_growl()
{
    g_fs.addSearchPath("../roms/growl.zip");

    load_audio("c74-12.ic62");

    sdram.load_data("c74-10-1.ic59", CPU_ROM_SDR_BASE + 1, 2);
    sdram.load_data("c74-08-1.ic61", CPU_ROM_SDR_BASE + 0, 2);
    sdram.load_data("c74-11.ic58", CPU_ROM_SDR_BASE + 0x80001, 2);
    sdram.load_data("c74-14.ic60", CPU_ROM_SDR_BASE + 0x80000, 2);
	
    sdram.load_data("c74-01.ic34", SCN0_ROM_SDR_BASE, 1);
    
    sdram.load_data("c74-04.ic28",  ADPCMA_ROM_SDR_BASE, 1);
    sdram.load_data("c74-05.ic29",  ADPCMB_ROM_SDR_BASE, 1);

    ddr_memory.load_data("c74-03.ic12", OBJ_DATA_DDR_BASE, 1);
    ddr_memory.load_data("c74-02.ic11", OBJ_DATA_DDR_BASE + 0x100000, 1);

    top->game = GAME_GROWL;
}

static void load_megab()
{
    g_fs.addSearchPath("../roms/megablst.zip");

    load_audio("c11-12.3");

    sdram.load_data("c11-07.55", CPU_ROM_SDR_BASE + 1, 2);
    sdram.load_data("c11-08.39", CPU_ROM_SDR_BASE + 0, 2);
    sdram.load_data("c11-06.54", CPU_ROM_SDR_BASE + 0x40001, 2);
    sdram.load_data("c11-11.38", CPU_ROM_SDR_BASE + 0x40000, 2);
	
    sdram.load_data("c11-05.58", SCN0_ROM_SDR_BASE, 1);
    
    sdram.load_data("c11-01.29",  ADPCMA_ROM_SDR_BASE, 1);
    sdram.load_data("c11-02.30",  ADPCMB_ROM_SDR_BASE, 1);

    ddr_memory.load_data("c11-03.32", OBJ_DATA_DDR_BASE, 2);
    ddr_memory.load_data("c11-04.31", OBJ_DATA_DDR_BASE + 1, 2);

    top->game = GAME_MEGAB;
}

static void load_driftout()
{
    g_fs.addSearchPath("../roms/driftout.zip");

    load_audio("do_50.rom");

    sdram.load_data("ic46.rom", CPU_ROM_SDR_BASE + 1, 2);
    sdram.load_data("ic45.rom", CPU_ROM_SDR_BASE + 0, 2);
	
    sdram.load_data("do_piv.rom",  PIVOT_ROM_SDR_BASE, 1);
    sdram.load_data("do_snd.rom",  ADPCMA_ROM_SDR_BASE, 1);

    ddr_memory.load_data("do_obj.rom", OBJ_DATA_DDR_BASE, 1);
    

    top->game = GAME_DRIFTOUT;
}

static void load_cameltry()
{
    g_fs.addSearchPath("../roms/cameltry.zip");

    load_audio("c38-08.bin");

    sdram.load_data("c38-11", CPU_ROM_SDR_BASE + 1, 2);
    sdram.load_data("c38-14", CPU_ROM_SDR_BASE + 0, 2);
	
    sdram.load_data("c38-02.bin", PIVOT_ROM_SDR_BASE, 1);
    sdram.load_data("c38-03.bin",  ADPCMA_ROM_SDR_BASE, 1);

    ddr_memory.load_data("c38-01.bin", OBJ_DATA_DDR_BASE, 1);

    top->game = GAME_CAMELTRY;
}


static void load_driftout_test()
{
    g_fs.addSearchPath("../testroms/build/driftout_test/driftout/");
    load_driftout();
    
    g_fs.addSearchPath("../roms/growl.zip");
    sdram.load_data("c74-01.ic34", SCN0_ROM_SDR_BASE, 1);
}

static void load_pulirula()
{
    g_fs.addSearchPath("../roms/pulirula.zip");

    load_audio("c98-14.rom");

    sdram.load_data("c98-12.rom", CPU_ROM_SDR_BASE + 1, 2);
    sdram.load_data("c98-16.rom", CPU_ROM_SDR_BASE + 0, 2);
    sdram.load_data("c98-06.rom", CPU_ROM_SDR_BASE + 0x80001, 2);
    sdram.load_data("c98-07.rom", CPU_ROM_SDR_BASE + 0x80000, 2);
		
    sdram.load_data("c98-04.rom",  SCN0_ROM_SDR_BASE, 1);

    sdram.load_data("c98-05.rom", PIVOT_ROM_SDR_BASE, 1);

    sdram.load_data("c98-01.rom",  ADPCMA_ROM_SDR_BASE, 1);

    ddr_memory.load_data("c98-02.rom", OBJ_DATA_DDR_BASE, 1);
    ddr_memory.load_data("c98-03.rom", OBJ_DATA_DDR_BASE + 0x100000, 1);

    top->game = GAME_PULIRULA;
}

static void load_ninjak()
{
    g_fs.addSearchPath("../roms/ninjak.zip");

    load_audio("c85-14.ic54");

    sdram.load_data("c85-10x.ic50", CPU_ROM_SDR_BASE + 1, 2);
    sdram.load_data("c85-13x.ic49", CPU_ROM_SDR_BASE + 0, 2);
    sdram.load_data("c85-07.ic48", CPU_ROM_SDR_BASE + 0x40001, 2);
    sdram.load_data("c85-06.ic47", CPU_ROM_SDR_BASE + 0x40000, 2);
		
    sdram.load_data("c85-03.ic65",  SCN0_ROM_SDR_BASE, 1);

    sdram.load_data("c85-04.ic31",  ADPCMA_ROM_SDR_BASE, 1);
    sdram.load_data("c85-05.ic33",  ADPCMB_ROM_SDR_BASE, 1);

    ddr_memory.load_data("c85-01.ic19", OBJ_DATA_DDR_BASE, 1);
    ddr_memory.load_data("c85-02.ic17", OBJ_DATA_DDR_BASE + 0x100000, 1);

    top->game = GAME_NINJAK;
}

static void load_thundfox()
{
    g_fs.addSearchPath("../roms/thundfox.zip");

    load_audio("c28-14.3");

    sdram.load_data("c28-13-1.51", CPU_ROM_SDR_BASE + 1, 2);
    sdram.load_data("c28-16-1.40", CPU_ROM_SDR_BASE + 0, 2);
    sdram.load_data("c28-08.50", CPU_ROM_SDR_BASE + 0x40001, 2);
    sdram.load_data("c28-07.39", CPU_ROM_SDR_BASE + 0x40000, 2);
		
    sdram.load_data("c28-02.61",  SCN0_ROM_SDR_BASE, 1);
    sdram.load_data("c28-01.63",  SCN1_ROM_SDR_BASE, 1);

    sdram.load_data("c28-06.41",  ADPCMA_ROM_SDR_BASE, 1);
    sdram.load_data("c28-05.42",  ADPCMB_ROM_SDR_BASE, 1);

    ddr_memory.load_data("c28-03.29", OBJ_DATA_DDR_BASE, 2);
    ddr_memory.load_data("c28-04.28", OBJ_DATA_DDR_BASE + 0x1, 2);

    top->game = GAME_THUNDFOX;
}


bool game_init(game_t game)
{
    g_fs.clearSearchPaths();

    switch(game)
    {
        case GAME_FINALB: load_finalb(); break;
        case GAME_QJINSEI: load_qjinsei(); break;
        case GAME_LIQUIDK: load_liquidk(); break;
        case GAME_DINOREX: load_dinorex(); break;
        case GAME_FINALB_TEST: load_finalb_test(); break;
        case GAME_QJINSEI_TEST: load_qjinsei_test(); break;
        case GAME_GROWL: load_growl(); break;
        case GAME_MEGAB: load_megab(); break;
        case GAME_DRIFTOUT: load_driftout(); break;
        case GAME_DRIFTOUT_TEST: load_driftout_test(); break;
        case GAME_CAMELTRY: load_cameltry(); break;
        case GAME_PULIRULA: load_pulirula(); break;
        case GAME_NINJAK: load_ninjak(); break;
        case GAME_THUNDFOX: load_thundfox(); break;
        default: return false;
    }

    return true;
}


