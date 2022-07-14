## Introduction

This is an attempt to have an open source reimplementation of PICO-8 fantasy console to be used on Desktop platforms but especially wherever you want to compile it.

It was born as an attempt to make PICO-8 games playable on OpenDingux devices (GCW0, RG350, ..).
It has now been extended to be compiled as a RetroArch core too.

## Implementation

The emulator is written in C++11 and embeds Lua source code (to allow extensions to the language that PICO-8 has). It has a SDL2.0 back-end but a SDL1.2 back-end wouldn't be hard to implement.

## Status

Currently much of the API is already working with good performance, even basic sound and music are working.

Many demos already work and even some full games.

- All graphics functions have been implemented but not all their subfeatures,
- All math functions have been implemented,
- Sound functions have been implemented together with an audio renderer stack but many effects still missing
- Common platform functions have been implemented
- Some Lua language extensions have been implemented
- Many quirks of the Lua extensions are implemented but some of most obscure things are still missing

Fixed arithmetic support is still missing.

## Building

If you want to build a libretro backend:

```
make
```

If you want to build a local binary you can run:

```
cmake .
make
```

# API Status
| __Graphics__ | implemented? | tests? | notes |
| --- | --- | --- | --- |
| `camera([x,] [y])` | ✔ | ✔ | |
| `circ(x, y, r, [col])` | ✔ |  | |
| `circfill(x, y, r, [col])` | ✔ | | |
| `clip([x,] [y,] [w,] [h])` | ✔ | | |
| `cls()` | ✔ | | |
| `color(col)` | ✔ | | |
| `cursor([x,] [y,] [col])` | ✔ | | |
| `fget(n, [f])` | ✔ | | |
| `fillp([pat])` |  | | |
| `fset(n, [f,] [v])` | ✔ | | |
| `line(x0, y0, x1, y1, [col])` | ✔ | | |
| `pal([c0,] [c1,] [p])` | ✔ | | |
| `palt([c,] [t])` | ✔ | | |
| `print(str, [x,] [y,] [col])` | ✔ | | |
| `pset(x, y, [c])` | ✔ | | |
| `rect(x0, y0, x1, y1, [col])` | ✔ | | |
| `rectfill(x0, y0, x1, y1, [col])` | ✔ | | |
| `spr(n, x, y, [w,] [h,] [flip_x,] [flip_y])` | ✔ | | |
| `sset(x, y, [c])` | ✔ | | |
| `sspr(sx, sy, sw, sh, dx, dy, [dw,] [dh,] [flip_x,] [flip_y])` | ✔ | | missing flip |
| __Input__ | | | |
| `btn([i,] [p])` | ✔ | | 1 player only |
| `btnp([i,] [p])` | ✔ | | not working as intended, 1 player only |
| __Math__ | | | |
| | | | `atan2` only one missing |
| __Tables__ | | | |
| | | | all functions implemented, not tested |
| __Map__ | | | |
| `map(cel_x, cel_y, sx, sy, cel_w, cel_h, [layer])` | ✔ |  | |
| `mget(x, y)` | ✔ | | |
| `mset(x, y, v)` | ✔ | | |
| __Cartridge__ | | | |
| `carddata` | ✔ | | just noop for now |
| `dget` | ✔ | | |
| `dset` | ✔ | | |
