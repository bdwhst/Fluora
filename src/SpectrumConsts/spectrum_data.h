#pragma once
#include <cuda.h>
namespace spec
{
    constexpr int nCIESamples = 471;
    extern const float CIE_lambda[nCIESamples];

    // Spectral Data
    extern const float CIE_X[nCIESamples];

    extern const float CIE_Y[nCIESamples];

    extern const float CIE_Z[nCIESamples];

    extern const float CIE_lambda[nCIESamples];


    extern const float CIE_Illum_A[];

    // CIE Illuminant D S basis functions
    constexpr int nCIES = 107;

    extern const float CIE_S_lambda[nCIES];

    extern const float CIE_S0[nCIES];

    extern const float CIE_S1[nCIES];

    extern const float CIE_S2[nCIES];

    extern const float CIE_Illum_D5000[];

    // Via https://gist.github.com/aforsythe/4df4e5377853df76a5a83a3c001c7eeb
    // with the critial bugfix:
    // <    cct = 6000
    // --
    // >    cct = 6000.
    extern const float ACES_Illum_D60[];

    extern const float CIE_Illum_D6500[] ;

    extern const float CIE_Illum_F1[] ;

    extern const float CIE_Illum_F2[];

    extern const float CIE_Illum_F3[];

    extern const float CIE_Illum_F4[];

    extern const float CIE_Illum_F5[];

    extern const float CIE_Illum_F6[];

    extern const float CIE_Illum_F7[];

    extern const float CIE_Illum_F8[];

    extern const float CIE_Illum_F9[];

    extern const float CIE_Illum_F10[];

    extern const float CIE_Illum_F11[];

    extern const float CIE_Illum_F12[];

    extern const float Ag_eta[];

    extern const float Ag_k[];

    extern const float Al_eta[];

    extern const float Al_k[];

    extern const float Au_eta[];

    extern const float Au_k[];

    extern const float Cu_eta[];

    extern const float Cu_k[];

    extern const float CuZn_eta[];

    extern const float CuZn_k[];

    extern const float MgO_eta[];

    extern const float MgO_k[];

    extern const float TiO2_eta[];

    extern const float TiO2_k[];

    // https://refractiveindex.info, public domain CC0:
    // https://creativecommons.org/publicdomain/zero/1.0/

    extern const float GlassBK7_eta[];

    extern const float GlassBAF10_eta[];

    extern const float GlassFK51A_eta[];

    extern const float GlassLASF9_eta[];

    extern const float GlassSF5_eta[] ;

    extern const float GlassSF10_eta[];

    extern const float GlassSF11_eta[];

    extern const float GlassSFake_eta[];

    // PhysLight code and measurements contributed by Anders Langlands and Luca Fascione
    // Copyright (c) 2020, Weta Digital, Ltd.
    // SPDX-License-Identifier: Apache-2.0
    extern const float canon_eos_100d_r[];
    extern const float canon_eos_100d_g[];
    extern const float canon_eos_100d_b[] ;

    extern const float canon_eos_1dx_mkii_r[] ;
    extern const float canon_eos_1dx_mkii_g[];
    extern const float canon_eos_1dx_mkii_b[] ;

    extern const float canon_eos_200d_r[] ;
    extern const float canon_eos_200d_g[] ;
    extern const float canon_eos_200d_b[];

    extern const float canon_eos_200d_mkii_r[];
    extern const float canon_eos_200d_mkii_g[];
    extern const float canon_eos_200d_mkii_b[] ;

    extern const float canon_eos_5d_r[];
    extern const float canon_eos_5d_g[];
    extern const float canon_eos_5d_b[];

    extern const float canon_eos_5d_mkii_r[] ;
    extern const float canon_eos_5d_mkii_g[] ;
    extern const float canon_eos_5d_mkii_b[] ;

    extern const float canon_eos_5d_mkiii_r[] ;
    extern const float canon_eos_5d_mkiii_g[] ;
    extern const float canon_eos_5d_mkiii_b[];

    extern const float canon_eos_5d_mkiv_r[];
    extern const float canon_eos_5d_mkiv_g[];
    extern const float canon_eos_5d_mkiv_b[];

    extern const float canon_eos_5ds_r[];
    extern const float canon_eos_5ds_g[] ;
    extern const float canon_eos_5ds_b[] ;

    extern const float canon_eos_m_r[];
    extern const float canon_eos_m_g[] ;
    extern const float canon_eos_m_b[] ;

    extern const float hasselblad_l1d_20c_r[];
    extern const float hasselblad_l1d_20c_g[];
    extern const float hasselblad_l1d_20c_b[] ;

    extern const float nikon_d810_r[];
    extern const float nikon_d810_g[];
    extern const float nikon_d810_b[];

    extern const float nikon_d850_r[];
    extern const float nikon_d850_g[];

    extern const float sony_ilce_6400_r[];
    extern const float sony_ilce_6400_g[];
    extern const float sony_ilce_6400_b[];
    extern const float sony_ilce_7m3_r[];
    extern const float sony_ilce_7m3_g[];
    extern const float sony_ilce_7m3_b[];
    extern const float sony_ilce_7rm3_r[];
    extern const float sony_ilce_7rm3_g[];
    extern const float sony_ilce_7rm3_b[];
    extern const float sony_ilce_9_r[];
    extern const float sony_ilce_9_g[];
    extern const float sony_ilce_9_b[];
}