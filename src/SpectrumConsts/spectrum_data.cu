#include "spectrum_data.h"
#include "spectrum_tables.inl"
#include <map>
#include "../spectrum.h"
#include "../utilities.h"
namespace spec
{
    std::map<std::string, SpectrumPtr> namedSpectra;

    DenselySampledSpectrum* x, * y, * z;
    __device__ DenselySampledSpectrum* dev_x, * dev_y, * dev_z;


    void init(Allocator alloc)
    {
        //todo: use cpu mem
        Allocator hostMemAlloc = Allocator(MainMemoryResourceBackend::getInstance());
        PiecewiseLinearSpectrum xpls(spec::nCIESamples, (float*)spec::CIE_lambda, (float*)spec::CIE_X, hostMemAlloc);
        PiecewiseLinearSpectrum ypls(spec::nCIESamples, (float*)spec::CIE_lambda, (float*)spec::CIE_Y, hostMemAlloc);
        PiecewiseLinearSpectrum zpls(spec::nCIESamples, (float*)spec::CIE_lambda, (float*)spec::CIE_Z, hostMemAlloc);

        x = alloc.new_object<DenselySampledSpectrum>(&xpls, alloc);
        y = alloc.new_object<DenselySampledSpectrum>(&ypls, alloc);
        z = alloc.new_object<DenselySampledSpectrum>(&zpls, alloc);
        cudaMemcpyToSymbol(dev_x, &x, sizeof(x));
        cudaMemcpyToSymbol(dev_y, &y, sizeof(y));
        cudaMemcpyToSymbol(dev_z, &z, sizeof(z));

        SpectrumPtr illuma = FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_A, true, alloc);
        SpectrumPtr illumd50 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_D5000, true, alloc);
        SpectrumPtr illumacesd60 =
            FROM_INTERLEAVED_FLT_ARRAY(ACES_Illum_D60, true, alloc);
        SpectrumPtr illumd65 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_D6500, true, alloc);
        SpectrumPtr illumf1 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F1, true, alloc);
        SpectrumPtr illumf2 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F2, true, alloc);
        SpectrumPtr illumf3 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F3, true, alloc);
        SpectrumPtr illumf4 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F4, true, alloc);
        SpectrumPtr illumf5 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F5, true, alloc);
        SpectrumPtr illumf6 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F6, true, alloc);
        SpectrumPtr illumf7 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F7, true, alloc);
        SpectrumPtr illumf8 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F8, true, alloc);
        SpectrumPtr illumf9 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F9, true, alloc);
        SpectrumPtr illumf10 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F10, true, alloc);
        SpectrumPtr illumf11 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F11, true, alloc);
        SpectrumPtr illumf12 =
            FROM_INTERLEAVED_FLT_ARRAY(CIE_Illum_F12, true, alloc);

        SpectrumPtr ageta = FROM_INTERLEAVED_FLT_ARRAY(Ag_eta, false, alloc);
        SpectrumPtr agk = FROM_INTERLEAVED_FLT_ARRAY(Ag_k, false, alloc);
        SpectrumPtr aleta = FROM_INTERLEAVED_FLT_ARRAY(Al_eta, false, alloc);
        SpectrumPtr alk = FROM_INTERLEAVED_FLT_ARRAY(Al_k, false, alloc);
        SpectrumPtr aueta = FROM_INTERLEAVED_FLT_ARRAY(Au_eta, false, alloc);
        SpectrumPtr auk = FROM_INTERLEAVED_FLT_ARRAY(Au_k, false, alloc);
        SpectrumPtr cueta = FROM_INTERLEAVED_FLT_ARRAY(Cu_eta, false, alloc);
        SpectrumPtr cuk = FROM_INTERLEAVED_FLT_ARRAY(Cu_k, false, alloc);
        SpectrumPtr cuzneta = FROM_INTERLEAVED_FLT_ARRAY(CuZn_eta, false, alloc);
        SpectrumPtr cuznk = FROM_INTERLEAVED_FLT_ARRAY(CuZn_k, false, alloc);
        SpectrumPtr mgoeta = FROM_INTERLEAVED_FLT_ARRAY(MgO_eta, false, alloc);
        SpectrumPtr mgok = FROM_INTERLEAVED_FLT_ARRAY(MgO_k, false, alloc);
        SpectrumPtr tio2eta = FROM_INTERLEAVED_FLT_ARRAY(TiO2_eta, false, alloc);
        SpectrumPtr tio2k = FROM_INTERLEAVED_FLT_ARRAY(TiO2_k, false, alloc);
        SpectrumPtr glassbk7eta =
            FROM_INTERLEAVED_FLT_ARRAY(GlassBK7_eta, false, alloc);
        SpectrumPtr glassbaf10eta =
            FROM_INTERLEAVED_FLT_ARRAY(GlassBAF10_eta, false, alloc);
        SpectrumPtr glassfk51aeta =
            FROM_INTERLEAVED_FLT_ARRAY(GlassFK51A_eta, false, alloc);
        SpectrumPtr glasslasf9eta =
            FROM_INTERLEAVED_FLT_ARRAY(GlassLASF9_eta, false, alloc);
        SpectrumPtr glasssf5eta =
            FROM_INTERLEAVED_FLT_ARRAY(GlassSF5_eta, false, alloc);
        SpectrumPtr glasssf10eta =
            FROM_INTERLEAVED_FLT_ARRAY(GlassSF10_eta, false, alloc);
        SpectrumPtr glasssf11eta =
            FROM_INTERLEAVED_FLT_ARRAY(GlassSF11_eta, false, alloc);

        SpectrumPtr glassfakeeta =
            FROM_INTERLEAVED_FLT_ARRAY(GlassSFake_eta, false, alloc);


        namedSpectra = {
        {"glass-BK7", glassbk7eta},
        {"glass-BAF10", glassbaf10eta},
        {"glass-FK51A", glassfk51aeta},
        {"glass-LASF9", glasslasf9eta},
        {"glass-F5", glasssf5eta},
        {"glass-F10", glasssf10eta},
        {"glass-F11", glasssf11eta},
        {"glass-Fake", glassfakeeta},

        {"metal-Ag-eta", ageta},
        {"metal-Ag-k", agk},
        {"metal-Al-eta", aleta},
        {"metal-Al-k", alk},
        {"metal-Au-eta", aueta},
        {"metal-Au-k", auk},
        {"metal-Cu-eta", cueta},
        {"metal-Cu-k", cuk},
        {"metal-CuZn-eta", cuzneta},
        {"metal-CuZn-k", cuznk},
        {"metal-MgO-eta", mgoeta},
        {"metal-MgO-k", mgok},
        {"metal-TiO2-eta", tio2eta},
        {"metal-TiO2-k", tio2k},

        {"stdillum-A", illuma},
        {"stdillum-D50", illumd50},
        {"stdillum-D65", illumd65},
        {"stdillum-F1", illumf1},
        {"stdillum-F2", illumf2},
        {"stdillum-F3", illumf3},
        {"stdillum-F4", illumf4},
        {"stdillum-F5", illumf5},
        {"stdillum-F6", illumf6},
        {"stdillum-F7", illumf7},
        {"stdillum-F8", illumf8},
        {"stdillum-F9", illumf9},
        {"stdillum-F10", illumf10},
        {"stdillum-F11", illumf11},
        {"stdillum-F12", illumf12},

        {"illum-acesD60", illumacesd60},

        {"canon_eos_100d_r",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_100d_r, false, alloc)},
        {"canon_eos_100d_g",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_100d_g, false, alloc)},
        {"canon_eos_100d_b",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_100d_b, false, alloc)},

        {"canon_eos_1dx_mkii_r",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_1dx_mkii_r, false, alloc)},
        {"canon_eos_1dx_mkii_g",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_1dx_mkii_g, false, alloc)},
        {"canon_eos_1dx_mkii_b",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_1dx_mkii_b, false, alloc)},

        {"canon_eos_200d_r",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_200d_r, false, alloc)},
        {"canon_eos_200d_g",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_200d_g, false, alloc)},
        {"canon_eos_200d_b",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_200d_b, false, alloc)},

        {"canon_eos_200d_mkii_r",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_200d_mkii_r, false, alloc)},
        {"canon_eos_200d_mkii_g",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_200d_mkii_g, false, alloc)},
        {"canon_eos_200d_mkii_b",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_200d_mkii_b, false, alloc)},

        {"canon_eos_5d_r",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_r, false, alloc)},
        {"canon_eos_5d_g",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_g, false, alloc)},
        {"canon_eos_5d_b",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_b, false, alloc)},

        {"canon_eos_5d_mkii_r",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_mkii_r, false, alloc)},
        {"canon_eos_5d_mkii_g",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_mkii_g, false, alloc)},
        {"canon_eos_5d_mkii_b",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_mkii_b, false, alloc)},

        {"canon_eos_5d_mkiii_r",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_mkiii_r, false, alloc)},
        {"canon_eos_5d_mkiii_g",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_mkiii_g, false, alloc)},
        {"canon_eos_5d_mkiii_b",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_mkiii_b, false, alloc)},

        {"canon_eos_5d_mkiv_r",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_mkiv_r, false, alloc)},
        {"canon_eos_5d_mkiv_g",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_mkiv_g, false, alloc)},
        {"canon_eos_5d_mkiv_b",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5d_mkiv_b, false, alloc)},

        {"canon_eos_5ds_r",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5ds_r, false, alloc)},
        {"canon_eos_5ds_g",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5ds_g, false, alloc)},
        {"canon_eos_5ds_b",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_5ds_b, false, alloc)},

        {"canon_eos_m_r",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_m_r, false, alloc)},
        {"canon_eos_m_g",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_m_g, false, alloc)},
        {"canon_eos_m_b",
         FROM_INTERLEAVED_FLT_ARRAY(canon_eos_m_b, false, alloc)},

        {"hasselblad_l1d_20c_r",
         FROM_INTERLEAVED_FLT_ARRAY(hasselblad_l1d_20c_r, false, alloc)},
        {"hasselblad_l1d_20c_g",
         FROM_INTERLEAVED_FLT_ARRAY(hasselblad_l1d_20c_g, false, alloc)},
        {"hasselblad_l1d_20c_b",
         FROM_INTERLEAVED_FLT_ARRAY(hasselblad_l1d_20c_b, false, alloc)},

        {"nikon_d810_r",
         FROM_INTERLEAVED_FLT_ARRAY(nikon_d810_r, false, alloc)},
        {"nikon_d810_g",
         FROM_INTERLEAVED_FLT_ARRAY(nikon_d810_g, false, alloc)},
        {"nikon_d810_b",
         FROM_INTERLEAVED_FLT_ARRAY(nikon_d810_b, false, alloc)},

        {"nikon_d850_r",
         FROM_INTERLEAVED_FLT_ARRAY(nikon_d850_r, false, alloc)},
        {"nikon_d850_g",
         FROM_INTERLEAVED_FLT_ARRAY(nikon_d850_g, false, alloc)},
        {"nikon_d850_b",
         FROM_INTERLEAVED_FLT_ARRAY(nikon_d850_b, false, alloc)},

        {"sony_ilce_6400_r",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_6400_r, false, alloc)},
        {"sony_ilce_6400_g",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_6400_g, false, alloc)},
        {"sony_ilce_6400_b",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_6400_b, false, alloc)},

        {"sony_ilce_7m3_r",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_7m3_r, false, alloc)},
        {"sony_ilce_7m3_g",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_7m3_g, false, alloc)},
        {"sony_ilce_7m3_b",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_7m3_b, false, alloc)},

        {"sony_ilce_7rm3_r",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_7rm3_r, false, alloc)},
        {"sony_ilce_7rm3_g",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_7rm3_g, false, alloc)},
        {"sony_ilce_7rm3_b",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_7rm3_b, false, alloc)},

        {"sony_ilce_9_r",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_9_r, false, alloc)},
        {"sony_ilce_9_g",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_9_g, false, alloc)},
        {"sony_ilce_9_b",
         FROM_INTERLEAVED_FLT_ARRAY(sony_ilce_9_b, false, alloc)} };

        checkCUDAError("spectrum init");
    }

    SpectrumPtr get_named_spectrum(const std::string& name)
    {
        auto iter = namedSpectra.find(name);
        if (iter != namedSpectra.end())
            return iter->second;
        return nullptr;
    }
}