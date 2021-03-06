#pragma once

#define PLATFORM_WIN32 0
#define PLATFORM_LIBRETRO 1

#define SOUND_ENABLED true

#if defined(__LIBRETRO__)
#define PLATFORM PLATFORM_LIBRETRO
#elif defined(_WIN32)
#define PLATFORM PLATFORM_WIN32
#endif

#define MOUSE_ENABLED false
#define TEST_MODE false

#define R8_OPTS_ENABLED true
#define R8_USE_LODE_PNG true

#if PLATFORM != PLATFORM_LIBRETRO

  #include "SDL.h"
  #define LOGD(x , ...) printf(x"\n", ## __VA_ARGS__)

  #if PLATFORM == PLATFORM_WIN32
    static constexpr int SCREEN_WIDTH = 240;
    static constexpr int SCREEN_HEIGHT = 240;

    #undef MOUSE_ENABLED
    #define MOUSE_ENABLED true

    static constexpr auto KEY_UP = SDLK_UP;
    static constexpr auto KEY_DOWN = SDLK_DOWN;
    static constexpr auto KEY_LEFT = SDLK_LEFT;
    static constexpr auto KEY_RIGHT = SDLK_RIGHT;

    static constexpr auto KEY_ACTION1_1 = SDLK_z;
    static constexpr auto KEY_ACTION1_2 = SDLK_x;
    static constexpr auto KEY_ACTION2_1 = SDLK_a;
    static constexpr auto KEY_ACTION2_2 = SDLK_s;

    static constexpr auto KEY_MUTE = SDLK_m;
    static constexpr auto KEY_PAUSE = SDLK_p;

    static constexpr auto KEY_NEXT_SCALER = SDLK_v;

    static constexpr auto KEY_MENU = SDLK_RETURN;
    static constexpr auto KEY_EXIT = SDLK_ESCAPE;
  #endif

#else

#define LOGD(x, ...)

#endif
