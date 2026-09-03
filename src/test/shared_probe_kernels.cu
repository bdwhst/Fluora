// CUDA registration TU for SharedHostTest's probe kernel: the same shared
// headers the host personality compiles, now under nvcc.
#include "../rhi/gpu_portable.h"
#include "../rhi/primitives_shared.h"
#include "../core/spectrum_shared.h"
#include "../core/bsdf_shared.h"
#include "../core/envmap_shared.h"
#include "../core/tonemap_shared.h"
#include "shared_probe.h"
#include "../rhi/rhi_cuda.h"

RHI_CUDA_REGISTER_KERNEL(shared_probe);
