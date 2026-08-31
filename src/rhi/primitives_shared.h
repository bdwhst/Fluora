#ifndef RHI_PRIMITIVES_SHARED_H
#define RHI_PRIMITIVES_SHARED_H
// Parameter block shared by the parallel-primitive kernels (primitives_gpu.h)
// and their host wrappers (rhi_algorithms.cpp). Plain unsigned ints only, so
// host/MSL layout agreement is trivial (invariant I-3). One struct serves all
// kernels; fields are ignored where irrelevant. Textually prepended to
// primitives_gpu.h at runtime MSL compile — keep self-contained.

#define PRIM_TILE 256u
#define PRIM_RADIX_BITS 4u
#define PRIM_RADIX_DIGITS 16u

struct PrimParams {
    unsigned int n;          // element count for this dispatch
    unsigned int numBlocks;  // threadgroups over the input (radix layout stride)
    unsigned int shift;      // radix: current digit shift (0,4,...,28)
    unsigned int pad;
};

#endif
