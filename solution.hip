#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <iomanip>
#include <bit>

#define HIP_CHECK(condition)                                                                  \
    do                                                                                        \
    {                                                                                         \
        hipError_t error = condition;                                                         \
        if(error != hipSuccess)                                                               \
        {                                                                                     \
            std::cerr << "Error " << hipGetErrorName(error) << '(' << error << ')' << ": "    \
                      << hipGetErrorString(error) << " in " << __func__ << " at " << __FILE__ \
                      << ':' << __LINE__ << '\n';                                             \
            exit(error);                                                                      \
        }                                                                                     \
    }                                                                                         \
    while(false)

using fp8 = __hip_fp8_e4m3_fnuz;
using bf16 = __hip_bfloat16;
using fp32 = float;
using fp32x4 = fp32 __attribute__((ext_vector_type(4)));
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using fp32x16 = fp32 __attribute__((ext_vector_type(16)));

struct cube {
    int m, n, k;
};

constexpr int warp_size = 64;
constexpr int cache_line_size = 128;

constexpr int m_block_size_multiple = 32;
constexpr int n_block_size_multiple = 32;
constexpr int k_block_size_multiple = 128;
constexpr int group_size = 128;

using register_type = uint32_t;

template <typename T, typename U>
__host__ __device__ __forceinline__
constexpr auto ceil_div(T a, U b) {
    return (a + b - 1) / b;
}

template<auto a, auto b>
__host__ __device__ __forceinline__
constexpr auto divide_exact() {
    static_assert(a % b == 0, "does not divide exact!");
    return a / b;
}

template<int inc, int add, typename seq>
struct multiply_by_inc;

template<int inc, int add, int... Is>
struct multiply_by_inc<inc, add, std::integer_sequence<int, Is...>> {
    using type = std::integer_sequence<int, (inc * Is + add)...>;
};

template<int I, int N, int inc, int... Is>
struct make_index_sequence_with_inc_helper {
    using type = typename multiply_by_inc<inc, I, std::make_integer_sequence<int, (N - I) / inc>>::type;
};

template<int I, int N, int inc>
using make_index_sequence_with_inc = typename make_index_sequence_with_inc_helper<I, N, inc>::type;

template<typename F, int... Is>
__host__ __device__ __forceinline__
void unroll_apply(F f, std::integer_sequence<int, Is...>) {
    (f.template operator()<Is>(), ...);
}

template<int I, int N, int inc = 1, typename F>
__host__ __device__ __forceinline__
void unroll(F f) {
    unroll_apply(f, make_index_sequence_with_inc<I, N, inc>{});
}

template<int I, typename F>
__host__ __device__ __forceinline__
void unroll1d(F f) {
    unroll<0, I>(f);
}

template<int I, int J, typename F>
__host__ __device__ __forceinline__
void unroll2d(F f) {
    unroll<0, I>([&]<int i> {
        unroll<0, J>([&]<int j> {
            f.template operator()<i, j>();
        });
    });
}

template<int I, int J, int K, typename F>
__host__ __device__ __forceinline__
void unroll3d(F f) {
    unroll<0, I>([&]<int i> {
        unroll<0, J>([&]<int j> {
            unroll<0, K>([&]<int k> {
                f.template operator()<i, j, k>();
            });
        });
    });
}

template <typename T, size_t N>
__device__ __forceinline__
void clear(T (&array)[N]) {
    unroll<0, N>([&]<int i> {
        array[i] = T{0};
    });
}

__device__ __forceinline__
bf16 bf16_from_fp32(float x) {
    // HIP's conversion function does all kinds of checks
    // for odd cases (NaN, subnormals) that we don't care about.
    // This generates absolutely abysmal code, so just truncate
    // the upper bits to get the right value like the
    // Opperpython intended.

    return std::bit_cast<bf16>(
        static_cast<uint16_t>(
            std::bit_cast<uint32_t>(x) >> 16
        )
    );
}

__device__ __forceinline__
fp32 fp32_from_fp8(fp8 x) {
    // HIP's conversion from fp8 to fp32 is slow on anything that
    // doesn't have the hardware instructions to do it. The software
    // method deals with special cases (NaN, subnormals), but we
    // don't care about that here since we don't expect those in
    // the input. Therefore, we can optimize the software implementation
    // by just shuffling the bits.
#ifdef __gfx942__
    // MI300 has an instruction for this.
    return float(x);
#else
    // Fallback case
    // TODO: Find a faster option
    return float(x);
#endif
}

template <typename F>
__device__ __host__ __forceinline__
void select(bool cond, F f) {
    if (cond) {
        f.template operator()<true>();
    } else {
        f.template operator()<false>();
    }
}

template<int dpp_ctrl, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = false>
__device__ __forceinline__
uint32_t shfl_dpp(uint32_t x) {
    return __builtin_amdgcn_update_dpp(x, x, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
}

template<int lane_stride>
__device__ __forceinline__
uint32_t transpose_4x4_f8_v2(uint32_t input) {
    static_assert(lane_stride < warp_size / 4, "lane stride is too large");
    static_assert((lane_stride & (lane_stride - 1)) == 0, "line stride has to be a power of 2");
    static_assert(lane_stride == 1, "DPP implementation requires lane_stride == 1");
    // input:
    // byte 0 1 2 3
    //      0 1 2 3 lane X
    //      4 5 6 7 lane X+lane_stride
    //      8 9 A B lane X+lane_stride
    //      C D E F lane X+lane_stride
    // swap pairs of lanes to get
    // 0 1 2 3
    // 4 5 6 7
    // C D E F
    // 8 9 A B
    // then use byteperm to swap pairs horizontally + merge the
    // two results to get 2x2 transposed
    // 0 4 2 6
    // 1 5 3 7
    // 8 C A E
    // 9 D B F
    const auto lane = __lane_id();
    const auto group_lane_index = lane / lane_stride % 4;
    const auto t2x2 = __builtin_amdgcn_perm(
        // __shfl_xor(input, lane_stride), // Swap pairs
        shfl_dpp<0b10'11'00'01>(input),
        input,
        group_lane_index % 2 == 0
            ? 0x06'02'04'00
            : 0x03'07'01'05
    );
    // Now swap groups of 2 horizontally, and do the
    // same byteperm operation except with groups of 2x2.
    // 0 0 1 1     0 0 2 2
    // 0 0 1 1 --> 0 0 2 2
    // 2 2 3 3     1 1 3 3
    // 2 2 3 3     1 1 3 3
    const auto t4x4 = __builtin_amdgcn_perm(
        // __shfl_xor(t2x2, lane_stride * 2), // Swap groups
        shfl_dpp<0b01'00'11'10>(t2x2),
        t2x2,
        group_lane_index < 2
            ? 0x05'04'01'00
            : 0x03'02'07'06
    );

    // input = (lane << 24) | (lane << 16) | (lane << 8) | lane;
    return t4x4;
}

__device__ uint8_t llvm_amdgcn_raw_buffer_load_b8(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i8");

__device__ uint16_t llvm_amdgcn_raw_buffer_load_b16(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i16");

__device__ uint32_t llvm_amdgcn_raw_buffer_load_b32(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ uint64_t llvm_amdgcn_raw_buffer_load_b64(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i64");

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i128");


__device__ void llvm_amdgcn_raw_buffer_store_b8(uint8_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i8");

__device__ void llvm_amdgcn_raw_buffer_store_b16(uint16_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i16");

__device__ void llvm_amdgcn_raw_buffer_store_b32(uint32_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ void llvm_amdgcn_raw_buffer_store_b64(uint64_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i64");

__device__ void llvm_amdgcn_raw_buffer_store_b128(__uint128_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i128");

enum class coherency {
	cache_all = 0,
	cache_global = 1,
	cache_stream = 2,
	non_temporal = 3
};

template <typename T>
struct buffer {
    struct buffer_resource {
        const void* ptr;
        uint32_t range;
        uint32_t config;
    };

	using elem_type = std::remove_cv_t<T>;

    buffer_resource rsrc;

    __device__ __forceinline__
    buffer(T* pointer, uint32_t size) {
        this->rsrc = buffer_resource{
            reinterpret_cast<const void*>(pointer),
            static_cast<uint32_t>(size * sizeof(elem_type)),
            // This is what CK uses, but what does it mean? It seems to mean DATA_FORMAT = 32 bit
            // We might be able to play with this later...
            0x00020000U
        };
    }

    template<size_t N, coherency c = coherency::cache_all>
    __device__ __forceinline__
    std::array<elem_type, N> load(
        uint32_t wave_offset,
        uint32_t thread_offset,
        bool in_bounds = true
    ) const {
        using result_type = std::array<elem_type, N>;
        constexpr const int load_size = sizeof(elem_type) * N;

		wave_offset *= sizeof(elem_type);
		thread_offset *= sizeof(elem_type);
        thread_offset = in_bounds ? thread_offset : 0xFFFF'FFFF;
        const int cc = static_cast<int>(c);

        if constexpr (load_size == 1) {
            return std::bit_cast<result_type>(llvm_amdgcn_raw_buffer_load_b8(
                std::bit_cast<i32x4>(this->rsrc),
                thread_offset,
                wave_offset,
                cc
            ));
        } else if constexpr (load_size == 2) {
            return std::bit_cast<result_type>(llvm_amdgcn_raw_buffer_load_b16(
                std::bit_cast<i32x4>(this->rsrc),
                thread_offset,
                wave_offset,
                cc
            ));
        } else if constexpr (load_size == 4) {
            return std::bit_cast<result_type>(llvm_amdgcn_raw_buffer_load_b32(
                std::bit_cast<i32x4>(this->rsrc),
                thread_offset,
                wave_offset,
                cc
            ));
        } else if constexpr (load_size == 8) {
            return std::bit_cast<result_type>(llvm_amdgcn_raw_buffer_load_b64(
                std::bit_cast<i32x4>(this->rsrc),
                thread_offset,
                wave_offset,
                cc
            ));
        } else if constexpr (load_size == 16) {
            return std::bit_cast<result_type>(llvm_amdgcn_raw_buffer_load_b128(
                std::bit_cast<i32x4>(this->rsrc),
                thread_offset,
                wave_offset,
                cc
            ));
        } else {
            static_assert(load_size == -1, "load size not implemented");
        }
    }

    template<coherency c = coherency::cache_all, size_t N>
    __device__ __forceinline__
    void store(
	    const std::array<elem_type, N>& data,
        uint32_t wave_offset,
        uint32_t thread_offset,
        bool in_bounds = true
    ) const {
        constexpr const int store_size = sizeof(elem_type) * N;

		wave_offset *= sizeof(elem_type);
		thread_offset *= sizeof(elem_type);
        thread_offset = in_bounds ? thread_offset : 0xFFFF'FFFF;
        const int cc = static_cast<int>(c);

        if constexpr (store_size == 1) {
            llvm_amdgcn_raw_buffer_store_b8(
	            std::bit_cast<uint8_t>(data),
                std::bit_cast<i32x4>(this->rsrc),
                thread_offset,
                wave_offset,
                cc
            );
        } else if constexpr (store_size == 2) {
            llvm_amdgcn_raw_buffer_store_b16(
	            std::bit_cast<uint16_t>(data),
                std::bit_cast<i32x4>(this->rsrc),
                thread_offset,
                wave_offset,
                cc
            );
        } else if constexpr (store_size == 4) {
            llvm_amdgcn_raw_buffer_store_b32(
	            std::bit_cast<uint32_t>(data),
                std::bit_cast<i32x4>(this->rsrc),
                thread_offset,
                wave_offset,
                cc
            );
        } else if constexpr (store_size == 8) {
            llvm_amdgcn_raw_buffer_store_b64(
	            std::bit_cast<uint64_t>(data),
                std::bit_cast<i32x4>(this->rsrc),
                thread_offset,
                wave_offset,
                cc
            );
        } else if constexpr (store_size == 16) {
            llvm_amdgcn_raw_buffer_store_b128(
	            std::bit_cast<__uint128_t>(data),
                std::bit_cast<i32x4>(this->rsrc),
                thread_offset,
                wave_offset,
                cc
            );
        } else {
            static_assert(store_size == -1, "store size not implemented");
        }
    }

	template<coherency c = coherency::cache_all>
    __device__ __forceinline__
    void store(
	    const elem_type& data,
        uint32_t wave_offset,
        uint32_t thread_offset,
        bool in_bounds = true
    ) const {
	    store<c>(
		    std::array<elem_type, 1>({data}),
		    wave_offset,
		    thread_offset,
		    in_bounds
	    );
    }
};

// A general fast global memory needs to be as follows:
// - Group of 16 bytes so that we can use a 128-bit load
// - The fast path requires entire cache lines to be loaded at once. CDNA3
//   cache line size is 128 bytes, so the ideal matrix size has an inner
//   dimension (block_tile_m; source is column major) of at least 128.
// - We want every thread to participate, so we need at least
//   block_size / (128 / 16) elements in the N direction.
//   block_size == 64 => 8
//   block_size == 256 => 32
//   block_size == 1024 => 128
// - We assume that the input is already aligned to 128 bits, and that the m stride
//   is also aligned such. m, n, k are all divisible by 64, so this should be fine.
//
//   ┌───┬───┬───┬───┐ ┌───┐
//   │0  │4  │8  │12 │ │   │ = 16 bytes
//   ├───┼───┼───┼───┤ └───┘
//   │1  │5  │9  │13 │
//   ├───┼───┼───┼───┤
//   │2  │6  │10 │14 │
//   ├───┼───┼───┼───┤
//   │3  │7  │11 │15 │
//   └───┴───┴───┴───┘
template<
    int block_size,
    int block_tile_m,
    int block_tile_n,
    typename load_type,
    bool fused_transpose
>
struct block_loader {
    static_assert(block_tile_n % sizeof(load_type) == 0, "tile inner dimension has to be divisible by load type size");

    constexpr static int total_loads = block_tile_m * block_tile_n / sizeof(load_type);
    static_assert(total_loads % block_size == 0, "total amount of loads needs to be divisible by the block size");

    constexpr static int loads_per_block_tile_n = block_tile_n / sizeof(load_type);
    constexpr static int loads_per_thread = total_loads / block_size;

    using frg_type = std::array<load_type, loads_per_thread>;

    __device__ __forceinline__
    static frg_type load_from_global(
        const fp8* block_tile_src,
        int m,
        int n,
        int block_off_m,
        int block_off_n
    ) {
        const auto offset = block_off_n + block_off_m * n;
        const auto size = m * n;
        const auto buf = buffer<const uint8_t>(reinterpret_cast<const uint8_t*>(block_tile_src), m * n);

        const int tid = threadIdx.x;

        frg_type result;

        unroll<0, loads_per_thread>([&]<int j> {
            const int i = j * block_size + tid;
            const int row = i / loads_per_block_tile_n;
            const int col = i % loads_per_block_tile_n;
            const int src_off = row * n + col * sizeof(load_type);

            const int src_row = row + block_off_m;
            const int src_col = col * sizeof(load_type) + block_off_n;
            const bool in_bounds = src_row < m && src_col < n;

            const auto word = buf.template load<sizeof(load_type)>(offset, src_off, in_bounds);
            result[j] = std::bit_cast<load_type>(word);
        });

        return result;
    }

    __device__ __forceinline__
    static void store_to_smem(
        fp8* block_tile_dst,
        frg_type thread_src,
        int m,
        int n,
        int block_off_m,
        int block_off_n
    ) {
        const int tid = threadIdx.x;

        // The buffer load allows us to omit bounds checking with loading,
        // but that then loads invalid elements. To fix that, we check
        // for each element whether its supposed to be in range, and if not
        // zero it out.

        unroll<0, loads_per_thread>([&]<int j> {
            const int i = j * block_size + tid;

            const int row = i / loads_per_block_tile_n;
            const int col = i % loads_per_block_tile_n;
            const int dst_off = row * (block_tile_n + 4) + col * sizeof(load_type);

            auto* dst = reinterpret_cast<load_type*>(&block_tile_dst[dst_off]);
            *dst = thread_src[j];
        });

        // __syncthreads();
        // if (blockIdx.x == 0 && tid == 0) {
        //     printf("%d %d\n", block_off_m, block_off_n);
        //     for (int im = 0; im < block_tile_m; ++im) {
        //         for (int in = 0; in < block_tile_n; ++in) {
        //             printf("%.2f ", float(block_tile_dst[im * (block_tile_n) + in]));
        //         }
        //         printf("\n");
        //     }
        //     printf("-----\n");
        // }
        // __syncthreads();
    }
};

template<
    int block_size,
    int block_tile_m,
    int block_tile_n,
    typename load_type
>
struct block_loader<block_size, block_tile_m, block_tile_n, load_type, true> {
    static_assert(block_tile_m % sizeof(load_type) == 0, "tile inner dimension has to be divisible by load type size");

    constexpr static int total_loads = block_tile_m * block_tile_n / sizeof(load_type);
    static_assert(total_loads % block_size == 0, "total amount of loads needs to be divisible by the block size");

    constexpr static int loads_per_block_tile_m = block_tile_m / sizeof(load_type);
    constexpr static int loads_per_thread = total_loads / block_size;

    constexpr static int group_size = 4; // Coincides with 4x4 transpose.

    using frg_type = std::array<load_type, loads_per_thread>;

    __device__ __forceinline__
    static frg_type load_from_global(
        const fp8* block_tile_src,
        int m,
        int n,
        int block_off_m,
        int block_off_n
    ) {
        const auto offset = block_off_n * m + block_off_m;
        const auto buf = buffer<const uint8_t>(reinterpret_cast<const uint8_t*>(block_tile_src), m * n);

        const int tid = threadIdx.x;

        frg_type result;

        unroll<0, loads_per_thread>([&]<int j> {
            const int i = j * block_size + tid;

            // swizzle the lanes such that groups of 4 columns are next to eachother.
            const int group_index = i % group_size;
            const int group_bank = i / group_size;
            const int row = group_bank % loads_per_block_tile_m;
            const int col = group_index + group_bank / loads_per_block_tile_m * group_size;
            const int src_off = row * sizeof(load_type) + col * m;

            const int src_row = row * sizeof(load_type) + block_off_m;
            const int src_col = col + block_off_n;
            const bool in_bounds = src_row < m && src_col < n;

            // result[j] = buf.template load<1>(0, src_off)[0];

            const auto word = buf.template load<sizeof(load_type)>(offset, src_off, in_bounds);
            result[j] = std::bit_cast<load_type>(word);
        });

        return result;
    }

    __device__ __forceinline__
    static void store_to_smem(
        fp8* block_tile_dst,
        frg_type thread_src,
        int m,
        int n,
        int block_off_m,
        int block_off_n
    ) {
        const int tid = threadIdx.x;

        // The buffer load allows us to omit bounds checking with loading,
        // but that then loads invalid elements. To fix that, we check
        // for each element whether its supposed to be in range, and if not
        // zero it out.

        unroll<0, loads_per_thread>([&]<int j> {
            const int i = j * block_size + tid;

            // swizzle the lanes such that groups of 4 columns are next to eachother.
            const int group_index = i % group_size;
            const int group_bank = i / group_size;
            const int row = group_bank % loads_per_block_tile_m;
            const int col = group_index + group_bank / loads_per_block_tile_m * group_size;

            auto* dst = reinterpret_cast<uint8_t*>(block_tile_dst);


            // Store transposed using 4x4 groups.
            const auto src = std::bit_cast<std::array<uint32_t, 4>>(thread_src[j]);
            constexpr int lane_stride = 1; //loads_per_block_tile_m;
            unroll<0, sizeof(load_type), sizeof(uint32_t)>([&]<int k> {
                const int dst_row = row * sizeof(load_type) + k;
                const int dst_col = col;
                auto v = src[k / sizeof(uint32_t)];
                v = transpose_4x4_f8_v2<lane_stride>(v);
                const auto group_lane_index = __lane_id() / lane_stride % 4;
                *reinterpret_cast<uint32_t*>(&dst[(dst_col - dst_col % 4) + (dst_row + group_lane_index) * (block_tile_n + 4)])
                    = v;
            });
        });
    }
};


template <int warp_tile_m_, int warp_tile_n_, typename T>
struct mfma_fragment {
    using type = T;

    constexpr static const int warp_tile_m = warp_tile_m_;
    constexpr static const int warp_tile_n = warp_tile_n_;

    constexpr static const int total_items = warp_tile_m * warp_tile_n;

    constexpr static const int items_per_thread = divide_exact<total_items, warp_size>();
    constexpr static const int registers_per_thread = divide_exact<items_per_thread * sizeof(T), sizeof(register_type)>();
    constexpr static const int bytes_per_thread = items_per_thread * sizeof(T);

    register_type regs[registers_per_thread];

    __device__ __forceinline__
    constexpr static mfma_fragment zero() {
        return mfma_fragment{{}};
    }

    __device__ __forceinline__
    std::array<T, items_per_thread>& thread_values() {
        return *reinterpret_cast<std::array<T, items_per_thread>*>(&this->regs);
    }

    __device__ __forceinline__
    const std::array<T, items_per_thread>& thread_values() const {
        return *reinterpret_cast<const std::array<T, items_per_thread>*>(&this->regs);
    }

    template<typename U>
    __device__ __forceinline__
    std::array<U, bytes_per_thread / sizeof(U)>& regs_as() {
        static_assert(bytes_per_thread % sizeof(U) == 0, "total bytes in thread fragment not divisible by U");
        return *reinterpret_cast<std::array<U, bytes_per_thread / sizeof(U)>*>(&this->regs);
    }

    template<typename U>
    __device__ __forceinline__
    const std::array<U, bytes_per_thread / sizeof(U)>& regs_as() const {
        static_assert(bytes_per_thread % sizeof(U) == 0, "total bytes in thread fragment not divisible by U");
        return *reinterpret_cast<const std::array<U, bytes_per_thread / sizeof(U)>*>(&this->regs);
    }

    __device__ __forceinline__
    void clear() {
        unroll<0, registers_per_thread>([&]<int i> {
            this->regs[i] = 0;
        });
    }

    static constexpr int first_item_of_reg(int reg) {
        return reg * sizeof(register_type);
    }
};

template <int m, int n, int k>
struct mfma_f32_f32_f32;

template <>
struct mfma_f32_f32_f32<16, 16, 4> {
    using frg_a_type = mfma_fragment<16, 4, float>;
    using frg_b_type = mfma_fragment<16, 4, float>;
    using frg_c_type = mfma_fragment<16, 16, float>;

    __device__ __forceinline__
    static void execute(
        frg_c_type& c,
        const frg_a_type& a,
        const frg_b_type& b
    ) {
        c = std::bit_cast<frg_c_type>(
            __builtin_amdgcn_mfma_f32_16x16x4f32(
                std::bit_cast<float>(a),
                std::bit_cast<float>(b),
                std::bit_cast<fp32x4>(c),
                0,
                0,
                0
            )
        );
    }
};

template <>
struct mfma_f32_f32_f32<32, 32, 2> {
    using frg_a_type = mfma_fragment<32, 2, float>;
    using frg_b_type = mfma_fragment<32, 2, float>;
    using frg_c_type = mfma_fragment<32, 32, float>;

    __device__ __forceinline__
    static void execute(
        frg_c_type& c,
        const frg_a_type& a,
        const frg_b_type& b
    ) {
        c = std::bit_cast<frg_c_type>(
            __builtin_amdgcn_mfma_f32_32x32x2f32(
                std::bit_cast<float>(a),
                std::bit_cast<float>(b),
                std::bit_cast<fp32x16>(c),
                0,
                0,
                0
            )
        );
    }
};

template <int m, int n, int k>
struct mfma_f32_f8_f8;

template <>
struct mfma_f32_f8_f8<16, 16, 32> {
    using frg_a_type = mfma_fragment<16, 32, fp8>;
    using frg_b_type = mfma_fragment<16, 32, fp8>;
    using frg_c_type = mfma_fragment<16, 16, float>;

    __device__ __forceinline__
    static void execute(
        frg_c_type& c,
        const frg_a_type& a,
        const frg_b_type& b
    ) {
#ifdef __HIP_DEVICE_COMPILE__
#ifdef __gfx942__
    c = std::bit_cast<frg_c_type>(
        __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            std::bit_cast<uint64_t>(a),
            std::bit_cast<uint64_t>(b),
            std::bit_cast<fp32x4>(c),
            0,
            0,
            0
        )
    );
#else
    // Fallback copied from CK, this way we can test on MI210.
    using fallback_mfma = mfma_f32_f32_f32<16, 16, 4>;
    const auto& a_values = a.thread_values();
    const auto& b_values = b.thread_values();
    unroll<0, frg_a_type::items_per_thread>([&]<int k> {
        const auto fallback_a = std::bit_cast<fallback_mfma::frg_a_type>(static_cast<float>(a_values[k]));
        const auto fallback_b = std::bit_cast<fallback_mfma::frg_b_type>(static_cast<float>(b_values[k]));
        fallback_mfma::execute(c, fallback_a, fallback_b);
    });
#endif
#endif
    }
};

template<>
struct mfma_f32_f8_f8<32, 32, 16> {
    using frg_a_type = mfma_fragment<32, 16, fp8>;
    using frg_b_type = mfma_fragment<32, 16, fp8>;
    using frg_c_type = mfma_fragment<32, 32, float>;

    __device__ __forceinline__
    static void execute(
        frg_c_type& c,
        const frg_a_type& a,
        const frg_b_type& b
    ) {
#ifdef __HIP_DEVICE_COMPILE__
#ifdef __gfx942__
    c = std::bit_cast<frg_c_type>(
        __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
            std::bit_cast<uint64_t>(a),
            std::bit_cast<uint64_t>(b),
            std::bit_cast<fp32x16>(c),
            0,
            0,
            0
        )
    );
#else
    // Fallback copied from CK, this way we can test on MI210.
    using fallback_mfma = mfma_f32_f32_f32<32, 32, 2>;
    const auto& a_values = a.thread_values();
    const auto& b_values = b.thread_values();
    unroll<0, frg_a_type::items_per_thread>([&]<int k> {
        const auto fallback_a = std::bit_cast<fallback_mfma::frg_a_type>(static_cast<float>(a_values[k]));
        const auto fallback_b = std::bit_cast<fallback_mfma::frg_b_type>(static_cast<float>(b_values[k]));
        fallback_mfma::execute(c, fallback_a, fallback_b);
    });
#endif
#endif
    }
};

// See CDNA3 manual, 7.1.4.1. Input layout
// The calculations here are obtained via the
// AMD matrix instruction calculator
// https://github.com/ROCm/amd_matrix_instruction_calculator
// There is a cleaner way to compute the mapping, but
// this is what the tool returns and I'm lazy. We only
// really need these two anyway for now.
template <typename mfma>
struct mfma_indexer;

template <>
struct mfma_indexer<mfma_f32_f8_f8<16, 16, 32>> {
    __device__ __forceinline__
    static constexpr std::pair<int, int> item_to_matrix_a(int lane, int item) {
        const int m = lane % 16;
        const int k = 8 * (lane / 16) + item;
        return {m, k};
    }

    __device__ __forceinline__
    static constexpr std::pair<int, int> item_to_matrix_b(int lane, int item) {
        const int n = lane % 16;
        const int k = 8 * (lane / 16) + item;
        return {n, k};
    }

    __device__ __forceinline__
    static constexpr std::pair<int, int> item_to_matrix_c(int lane, int item) {
        const int m = 4 * (lane / 16) + item;
        const int n = lane % 16;
        return {m, n};
    }
};

template <>
struct mfma_indexer<mfma_f32_f8_f8<32, 32, 16>> {
    __device__ __forceinline__
    static constexpr std::pair<int, int> item_to_matrix_a(int lane, int item) {
        const int m = lane % 32;
        const int k = 8 * (lane / 32) + item;
        return {m, k};
    }

    __device__ __forceinline__
    static constexpr std::pair<int, int> item_to_matrix_b(int lane, int item) {
        const int n = lane % 32;
        const int k = 8 * (lane / 32) + item;
        return {n, k};
    }

    __device__ __forceinline__
    static constexpr std::pair<int, int> item_to_matrix_c(int lane, int item) {
        const int n = 8 * (item / 4) % 32 + 4 * (lane / 32) + (item % 4);
        const int m = lane % 32;
        return {n, m};
    }
};

template <
    int block_size,
    cube block_tile,
    cube warp_tile,
    typename load_type,
    int pipeline_depth = 1,
    bool fused_transpose = true,
    bool streamk
>
__global__ __launch_bounds__(block_size)
void kernel(
    const fp8* a,
    const fp8* b,
    const fp32* as,
    const fp32* bs,
    bf16* c,
    int m,
    int n,
    int k,
    int streamk_stride,
    fp32* streamk_tmp,
    int* streamk_counter
) {
    constexpr int warps = block_size / warp_size;

    constexpr int block_tile_m = block_tile.m;
    constexpr int block_tile_n = block_tile.n;
    constexpr int block_tile_k = block_tile.k;

    //static_assert(block_tile_m <= 128, "block tile M must be smaller than blockwise block size");
    //static_assert(block_tile_n <= 128, "block tile N must be smaller than blockwise block size");
    //static_assert(block_tile_k <= 128, "block tile K must be smaller than blockwise block size");

    constexpr int warp_tile_m = warp_tile.m;
    constexpr int warp_tile_n = warp_tile.n;
    constexpr int warp_tile_k = warp_tile.k;

    constexpr int c_total = block_tile_m * block_tile_n;

    static_assert(block_tile_m * block_tile_k % block_size == 0, "size(block_tile_a) has to be divisible by the block size");
    static_assert(block_tile_n * block_tile_k % block_size == 0, "size(block_tile_b) has to be divisible by the block size");
    static_assert(c_total % block_size == 0, "size(block_tile_c) as to be divisible by the block size");

    static_assert(block_tile_m % warp_tile_m == 0, "warp tile M has to divide block tile M");
    static_assert(block_tile_n % warp_tile_n == 0, "warp tile N has to divide block tile N");
    static_assert(block_tile_k % warp_tile_k == 0, "warp tile K has to divide block tile K");

    constexpr int warp_tiles_m = block_tile_m / warp_tile_m;
    constexpr int warp_tiles_n = block_tile_n / warp_tile_n;
    constexpr int warp_tiles_k = block_tile_k / warp_tile_k;
    constexpr int warp_tiles_mn = warp_tiles_m * warp_tiles_n;

	constexpr int intra_warp_tiles_m = warps == 1 ? 1 : 2;
	constexpr int intra_warp_tiles_n = warps == 1 || warps == 2 ? 1 : warps == 4 ? 2 : 4;
	constexpr int intra_warp_tile_m = divide_exact<warp_tiles_m, intra_warp_tiles_m>();
	constexpr int intra_warp_tile_n = divide_exact<warp_tiles_n, intra_warp_tiles_n>();

    // number of warp tiles has to be divisible by the number of warps
    constexpr int warp_tiles_mn_per_warp = divide_exact<warp_tiles_mn, warps>();
	static_assert(intra_warp_tile_m * intra_warp_tile_n == warp_tiles_mn_per_warp, "sanity check");
	static_assert(intra_warp_tiles_m * intra_warp_tiles_n == warps, "sanity check");

    constexpr int warp_tile_mn_items = warp_tile_m * warp_tile_n;

    using mfma = mfma_f32_f8_f8<warp_tile_m, warp_tile_n, warp_tile_k>;
    using frg_a_type = typename mfma::frg_a_type;
    using frg_b_type = typename mfma::frg_b_type;
    using frg_c_type = typename mfma::frg_c_type;
    using indexer = mfma_indexer<mfma>;

    using block_loader_a = block_loader<block_size, block_tile_m, block_tile_k, load_type, fused_transpose>;
    using block_loader_b = block_loader<block_size, block_tile_n, block_tile_k, load_type, fused_transpose>;

    constexpr int c_per_thread = c_total / block_size;
    static_assert(c_per_thread == warp_tiles_mn_per_warp * frg_c_type::items_per_thread, "sanity check");

    const int tid = threadIdx.x;

    const int warp = tid / warp_size;
    const int lane = __lane_id();

    // Swap direction to make bounds checking be first
    const int ti_m = blockIdx.y;
    const int ti_n = blockIdx.z;

	const int streamk_idx = blockIdx.x;
    const int k_start = streamk ? streamk_idx * streamk_stride : 0;
    const int k_end = streamk ? min(k, (streamk_idx + 1) * streamk_stride) : k;

	const int intra_warp_tile_im = warp % intra_warp_tiles_m;
	const int intra_warp_tile_in = warp / intra_warp_tiles_m;
	const int intra_warp_tile_om = intra_warp_tile_im * divide_exact<warp_tiles_m, intra_warp_tiles_m>();
	const int intra_warp_tile_on = intra_warp_tile_in * divide_exact<warp_tiles_n, intra_warp_tiles_n>();

    int sn = ceil_div(n, 128);

    // frg_c_type block_c_frg[warp_tiles_mn_per_warp];
    frg_c_type block_c_frg[intra_warp_tile_m][intra_warp_tile_n];
    unroll2d<intra_warp_tile_m, intra_warp_tile_n>([&]<int i, int j> {
        block_c_frg[i][j].clear();
    });

    __shared__ fp8 block_tile_a[block_tile_m * (block_tile_k + 4)];
    __shared__ fp8 block_tile_b[block_tile_n * (block_tile_k + 4)];
    __shared__ float block_tile_scaling[block_size];

    const auto load_input = [&](int i) {
        float frg_as;
        if (tid < 128) {
            frg_as = as[i / 128 * m + ti_m * block_tile_m + tid];
        }
        float frg_bs = bs[i / 128 * sn + ti_n * block_tile_n / 128];
        const auto frg_a = block_loader_a::load_from_global(
            a, m, k, ti_m * block_tile_m, i
        );
        const auto frg_b = block_loader_b::load_from_global(
            b, n, k, ti_n * block_tile_n, i
        );
        return std::make_tuple(frg_a, frg_b, frg_as, frg_bs);
    };

    const auto store_input = [&](int i, const auto& frg_a, const auto& frg_b, const auto& frg_as, const auto& frg_bs) {
        block_loader_a::store_to_smem(block_tile_a, frg_a, m, k, ti_m * block_tile_m, i);
        block_loader_b::store_to_smem(block_tile_b, frg_b, n, k, ti_n * block_tile_n, i);
        block_tile_scaling[tid] = frg_as * frg_bs;
    };

    frg_c_type warp_frg_c[intra_warp_tile_m][intra_warp_tile_n];

    const auto mma = [&]<bool first, bool last>(int i) {
        frg_a_type frg_a[intra_warp_tile_m][intra_warp_tile_n][warp_tiles_k];
        unroll2d<intra_warp_tile_m, intra_warp_tile_n>([&]<int iwtm, int iwtn> {
            const int wtm = iwtm * intra_warp_tiles_m + intra_warp_tile_im;
            const int wtn = iwtn * intra_warp_tiles_n + intra_warp_tile_in;
            // const int wtm = intra_warp_tile_om + iwtm;
            // const int wtn = intra_warp_tile_on + iwtn;

			const fp8* base = &block_tile_a[wtm * warp_tile_m * (block_tile_k + 4)];

            unroll<0, block_tile_k, warp_tile_k>([&]<int wtk_base> {
                constexpr int wtk = wtk_base / warp_tile_k;
                unroll<0, frg_a_type::registers_per_thread>([&]<int mk_reg> {
                    const auto [mfma_im, mfma_ik] = indexer::item_to_matrix_a(lane, frg_a_type::first_item_of_reg(mk_reg));
                    // const int bim = wtm * warp_tile_m + mfma_im;
                    const int bik = wtk_base + mfma_ik;
                    const int src_idx = bik + mfma_im * (block_tile_k + 4);
                    *reinterpret_cast<uint32_t*>(&frg_a[iwtm][iwtn][wtk].regs[mk_reg]) =
                        *reinterpret_cast<const uint32_t*>(&base[src_idx]);
                });
            });
        });

        frg_b_type frg_b[intra_warp_tile_m][intra_warp_tile_n][warp_tiles_k];
        unroll2d<intra_warp_tile_m, intra_warp_tile_n>([&]<int iwtm, int iwtn> {
            const int wtm = iwtm * intra_warp_tiles_m + intra_warp_tile_im;
            const int wtn = iwtn * intra_warp_tiles_n + intra_warp_tile_in;

            // const int wtm = intra_warp_tile_om + iwtm;
            // const int wtn = intra_warp_tile_on + iwtn;

			const fp8* base = &block_tile_b[wtn * warp_tile_n * (block_tile_k + 4)];

            unroll<0, block_tile_k, warp_tile_k>([&]<int wtk_base> {
                constexpr int wtk = wtk_base / warp_tile_k;
                unroll<0, frg_b_type::registers_per_thread>([&]<int nk_reg> {
                    const auto [mfma_in, mfma_ik] = indexer::item_to_matrix_b(lane, frg_b_type::first_item_of_reg(nk_reg));
                    // const int bin = wtn * warp_tile_n + mfma_in;
                    const int bik = wtk_base + mfma_ik;
                    const int src_idx = bik + mfma_in * (block_tile_k + 4);
                    *reinterpret_cast<uint32_t*>(&frg_b[iwtm][iwtn][wtk].regs[nk_reg]) =
                        *reinterpret_cast<const uint32_t*>(&base[src_idx]);
                });
            });
        });

        if constexpr (first) {
            unroll2d<intra_warp_tile_m, intra_warp_tile_n>([&]<int i, int j> {
                warp_frg_c[i][j].clear();
            });
        }

		// if constexpr (pipeline_depth == 2) {
	    //     __builtin_amdgcn_sched_barrier(0);
	    //     __builtin_amdgcn_s_setprio(1);
	    //     __builtin_amdgcn_sched_barrier(0);
        // }

        unroll3d<intra_warp_tile_m, intra_warp_tile_n, warp_tiles_k>([&]<int iwtm, int iwtn, int wtk> {
            mfma::execute(
                warp_frg_c[iwtm][iwtn],
                frg_a[iwtm][iwtn][wtk],
                frg_b[iwtm][iwtn][wtk]
            );
        });

        if constexpr (last) {
            unroll3d<intra_warp_tile_m, intra_warp_tile_n, frg_c_type::items_per_thread>([&]<int iwtm, int iwtn, int mn_item> {
                const int wtm = iwtm * intra_warp_tiles_m + intra_warp_tile_im;
                const int wtn = iwtn * intra_warp_tiles_n + intra_warp_tile_in;

                // const int wtm = intra_warp_tile_om + iwtm;
                // const int wtn = intra_warp_tile_on + iwtn;

                const auto [mfma_im, mfma_in] = indexer::item_to_matrix_c(lane, mn_item);
                const int bim = wtm * warp_tile_m + mfma_im;

                block_c_frg[iwtm][iwtn].thread_values()[mn_item] += warp_frg_c[iwtm][iwtn].thread_values()[mn_item]
                    * block_tile_scaling[bim];
            });
        }

		// if constexpr (pipeline_depth == 2) {
	    //     __builtin_amdgcn_sched_barrier(0);
	    //     __builtin_amdgcn_s_setprio(0);
	    //     __builtin_amdgcn_sched_barrier(0);
        // }
    };

    constexpr int group_tiles = divide_exact<group_size, block_tile_k>();
    constexpr int last_group_tile = group_tiles - 1;

    if constexpr (pipeline_depth == 0) {
        // Regular main loop
        // for (int i = k_start; i < k_end; i += block_tile_k) {
        //     const auto [frg_a, frg_b, frg_as, frg_bs] = load_input(i);
        //     store_input(i, frg_a, frg_b, frg_as, frg_bs);
        //     __syncthreads();
        //     mma(i);
        //     __syncthreads();
        // }

        int i = k_start;
        while (i < k_end) {
            unroll1d<group_tiles>([&]<int j> {
                const auto [frg_a, frg_b, frg_as, frg_bs] = load_input(i);
                store_input(i, frg_a, frg_b, frg_as, frg_bs);
                __syncthreads();
                mma.template operator()<j == 0, j == last_group_tile>(i);
                __syncthreads();

                i += block_tile_k;
            });
        }
    } else if constexpr (pipeline_depth == 1) {
        // Pipelined main loop with depth 1

        // int i = k_start;
        // auto [frg_a, frg_b, frg_as, frg_bs] = load_input(i);
        // store_input(i, frg_a, frg_b, frg_as, frg_bs);

        // for (; i + block_tile_k < k_end; i += block_tile_k) {
        //     std::tie(frg_a, frg_b, frg_as, frg_bs) = load_input(i + block_tile_k);
        //     __syncthreads();
        //     mma(i);
        //     __syncthreads();
        //     store_input(i + block_tile_k, frg_a, frg_b, frg_as, frg_bs);
        // }
        // __syncthreads();
        // mma(i);

        int i = k_start;
        auto [frg_a, frg_b, frg_as, frg_bs] = load_input(i);
        store_input(i, frg_a, frg_b, frg_as, frg_bs);
        while (i + group_size < k_end) {
            unroll1d<group_tiles>([&]<int j> {
                std::tie(frg_a, frg_b, frg_as, frg_bs) = load_input(i + block_tile_k);
                __syncthreads();
                mma.template operator()<j == 0, j == last_group_tile>(i);
                __syncthreads();
                store_input(i + block_tile_k, frg_a, frg_b, frg_as, frg_bs);
                i += block_tile_k;
            });
        }

        unroll1d<group_tiles>([&]<int j> {
            constexpr bool last = j == group_tiles - 1;
            if (!last) {
                std::tie(frg_a, frg_b, frg_as, frg_bs) = load_input(i + block_tile_k);
            }
            __syncthreads();
            mma.template operator()<j == 0, j == last_group_tile>(i);
            if (!last) {
                __syncthreads();
                store_input(i + block_tile_k, frg_a, frg_b, frg_as, frg_bs);
            }
            i += block_tile_k;
        });
    } else if constexpr (pipeline_depth == 2) {
        // Pipelined main loop with depth 2

        // int i = k_start;
        // auto [frg_a, frg_b, frg_as, frg_bs] = load_input(i);
        // store_input(i, frg_a, frg_b, frg_as, frg_bs);

        // // Prefetch next stage
        // auto [frg_a1, frg_b1, frg_as1, frg_bs1] = load_input(i + block_tile_k);

        // for(; i + block_tile_k + block_tile_k < k_end; i += block_tile_k) {
        //     std::tie(frg_a, frg_b, frg_as, frg_bs) = load_input(i + block_tile_k + block_tile_k);
        //     __syncthreads();
        //     mma(i);
        //     __syncthreads();
        //     store_input(i + block_tile_k, frg_a1, frg_b1, frg_as1, frg_bs1);
        //     frg_a1 = frg_a;
        //     frg_b1 = frg_b;
        //     frg_as1 = frg_as;
        //     frg_bs1 = frg_bs;
        // }
        // __syncthreads();
        // mma(i);
        // __syncthreads();
        // if (i + block_tile_k < k_end) {
        //     store_input(i + block_tile_k, frg_a1, frg_b1, frg_as1, frg_bs1);
        //     __syncthreads();
        //     mma(i + block_tile_k);
        // }

        int i = k_start;
        auto [frg_a, frg_b, frg_as, frg_bs] = load_input(i);
        store_input(i, frg_a, frg_b, frg_as, frg_bs);

        // Prefetch next stage
        auto [frg_a1, frg_b1, frg_as1, frg_bs1] = load_input(i + block_tile_k);

        while (i + group_size + group_size < k_end) {
            unroll1d<group_tiles>([&]<int j> {
                std::tie(frg_a, frg_b, frg_as, frg_bs) = load_input(i + block_tile_k + block_tile_k);
                __syncthreads();
                mma.template operator()<j == 0, j == last_group_tile>(i);
                __syncthreads();
                store_input(i + block_tile_k, frg_a1, frg_b1, frg_as1, frg_bs1);
                frg_a1 = frg_a;
                frg_b1 = frg_b;
                frg_as1 = frg_as;
                frg_bs1 = frg_bs;

                i += block_tile_k;
            });
        }

        unroll2d<2, group_tiles>([&]<int _, int j> {
            if (i + block_tile_k + block_tile_k < k_end) {
                std::tie(frg_a, frg_b, frg_as, frg_bs) = load_input(i + block_tile_k + block_tile_k);
            }
            __syncthreads();
            mma.template operator()<j == 0, j == last_group_tile>(i);
            __syncthreads();
            if (i + block_tile_k < k_end) {
                store_input(i + block_tile_k, frg_a1, frg_b1, frg_as1, frg_bs1);
            }
            frg_a1 = frg_a;
            frg_b1 = frg_b;
            frg_as1 = frg_as;
            frg_bs1 = frg_bs;

            i += block_tile_k;
        });
    } else {
	    static_assert(pipeline_depth >= 0 && pipeline_depth <= 2, "invalid pipeline depth");
    }

    // Write out C
    // This should access tile C in the same order as in the matmul loop!

    const int im_base = ti_m * block_tile_m;
    const int in_base = ti_n * block_tile_n;

	if constexpr (!streamk) {
		auto buf = buffer<bf16>(c, m * n);
	    unroll3d<intra_warp_tile_m, intra_warp_tile_n, frg_c_type::items_per_thread>([&]<int iwtm, int iwtn, int mn_item> {
	        const auto [mfma_im, mfma_in] = indexer::item_to_matrix_c(lane, mn_item);

            const int wtm = iwtm * intra_warp_tiles_m + intra_warp_tile_im;
            const int wtn = iwtn * intra_warp_tiles_n + intra_warp_tile_in;
	        // const int wtm = intra_warp_tile_om + iwtm;
	        // const int wtn = intra_warp_tile_on + iwtn;

	        const int bim = wtm * warp_tile_m + mfma_im;
	        const int bin = wtn * warp_tile_n + mfma_in;

	        const int im = im_base + bim;
	        const int in = in_base + bin;

	        const auto val = bf16_from_fp32(block_c_frg[iwtm][iwtn].thread_values()[mn_item]);

	        buf.store<coherency::non_temporal>(val, 0, im * n + in, im < m && in < n);
	    });
    } else {
	    if (streamk_idx == 0) {
		    unroll3d<intra_warp_tile_m, intra_warp_tile_n, frg_c_type::items_per_thread>([&]<int iwtm, int iwtn, int mn_item> {
		        const auto [mfma_im, mfma_in] = indexer::item_to_matrix_c(lane, mn_item);

                const int wtm = iwtm * intra_warp_tiles_m + intra_warp_tile_im;
                const int wtn = iwtn * intra_warp_tiles_n + intra_warp_tile_in;
		        // const int wtm = intra_warp_tile_om + iwtm;
		        // const int wtn = intra_warp_tile_on + iwtn;

		        const int bim = wtm * warp_tile_m + mfma_im;
		        const int bin = wtn * warp_tile_n + mfma_in;

		        const int im = im_base + bim;
		        const int in = in_base + bin;

				// Note: always in bounds
				__hip_atomic_store(&streamk_tmp[im * n + in], block_c_frg[iwtm][iwtn].thread_values()[mn_item],  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
		    });

            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
            // Wait until all vmem operations complete (s_waitcnt vmcnt(0))
            __builtin_amdgcn_s_waitcnt(/*vmcnt*/ 0 | (/*exp_cnt*/ 0x7 << 4) | (/*lgkmcnt*/ 0xf << 8));

			// Signal that the block is written.
			__syncthreads();
		    if (tid == 0) {
			    // atomicAdd(&streamk_counter[blockIdx.z * gridDim.y + blockIdx.y], 1);
                __hip_atomic_store(&streamk_counter[blockIdx.z * gridDim.y + blockIdx.y], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
		    }
	    } else {
		    // Poll until other block is done.
            while (true) {
                // int x = atomicAdd(&streamk_counter[blockIdx.z * gridDim.y + blockIdx.y], 0);
                int x = __hip_atomic_load(&streamk_counter[blockIdx.z * gridDim.y + blockIdx.y], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                if (x == 1) {
                    break;
                }
		    }

		    __syncthreads();

			auto buf = buffer<bf16>(c, m * n);
		    unroll3d<intra_warp_tile_m, intra_warp_tile_n, frg_c_type::items_per_thread>([&]<int iwtm, int iwtn, int mn_item> {
		        const auto [mfma_im, mfma_in] = indexer::item_to_matrix_c(lane, mn_item);

                const int wtm = iwtm * intra_warp_tiles_m + intra_warp_tile_im;
                const int wtn = iwtn * intra_warp_tiles_n + intra_warp_tile_in;
    	        // const int wtm = intra_warp_tile_om + iwtm;
    	        // const int wtn = intra_warp_tile_on + iwtn;

		        const int bim = wtm * warp_tile_m + mfma_im;
		        const int bin = wtn * warp_tile_n + mfma_in;

		        const int im = im_base + bim;
		        const int in = in_base + bin;

		        const auto val = bf16_from_fp32(
			        block_c_frg[iwtm][iwtn].thread_values()[mn_item]
			        + __hip_atomic_load(&streamk_tmp[im * n + in], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
		        );

		        buf.store<coherency::non_temporal>(val, 0, im * n + in, im < m && in < n);
		    });
	    }
    }
}

__device__
void transpose_matrix(
    fp8* output,
    const fp8* input,
    int m,
    int n,
    int block_x,
    int block_y
) {
	auto in = buffer<const uint8_t>(reinterpret_cast<const uint8_t*>(input), m * n);
	auto out = buffer<const uint8_t>(reinterpret_cast<const uint8_t*>(output), n * m);

	const auto tid = threadIdx.x;
	const int items_per_thread = 16;

	__shared__ uint8_t tmp[68 * 64];

	const auto bm = block_x * 64;
	const auto bn = block_y * 64;

	{
		const int i = tid * items_per_thread;
		const int o = bm + bn * m;
		const int x = i / 64;
		const int y = i % 64;

		const auto items = in.load<items_per_thread, coherency::non_temporal>(o, x * m + y);
		*reinterpret_cast<__uint128_t*>(&tmp[x * 68 + y]) = std::bit_cast<__uint128_t>(items);
	}

	__syncthreads();

	{
		const int i = tid * items_per_thread;
		const int o = bm * n + bn;
		const int x = i % 64;
		const int y = i / 64;

		std::array<uint8_t, items_per_thread> items;
		for (int j = 0; j < items_per_thread; ++j) {
			const int bx = (i + j) / 64;
			const int by = (i + j) % 64;
			items[j] = tmp[bx + by * 68];
		}

		out.store<coherency::cache_all>(
			items,
			o,
			x + y * n
		);
	}
}

__global__ __launch_bounds__(256)
void transpose_faster_fused(
    fp8* output_a,
    fp8* output_b,
    const fp8* input_a,
    const fp8* input_b,
    int m,
    int n,
    int k,
    int m_blocks
) {
	const int x = blockIdx.x;
	const int y = blockIdx.y;

	if (x < m_blocks) {
		transpose_matrix(
			output_a,
			input_a,
			m,
			k,
			x,
			y
		);
	} else {
		transpose_matrix(
			output_b,
			input_b,
			n,
			k,
			x - m_blocks,
			y
		);
	}
}

fp8* tmp_aT = nullptr;
fp8* tmp_bT = nullptr;

int prev_m = 0;
int prev_n = 0;
int prev_k = 0;

fp32* streamk_tmp = nullptr;
int* streamk_counter = nullptr;

template<
    int block_size,
    cube block_tile,
    cube warp_tile,
    int pipeline_depth = 1,
    bool fused_transpose = true,
    bool streamk = false,
    typename load_type = __uint128_t
>
void run_kernel(
    const fp8* a,
    const fp8* b,
    const fp32* as,
    const fp32* bs,
    bf16* c,
    int m,
    int n,
    int k,
    bool measure
) {
    const int grid_m = ceil_div(m, block_tile.m);
    const int grid_n = ceil_div(n, block_tile.n);

    if (!fused_transpose && (prev_m != m || prev_k != k || prev_n != n)) {
	    if (tmp_aT) {
		    HIP_CHECK(hipFree(tmp_aT));
		    HIP_CHECK(hipFree(tmp_bT));
	    }

        HIP_CHECK(hipMalloc(&tmp_aT, m * k * sizeof(fp8)));
        HIP_CHECK(hipMalloc(&tmp_bT, n * k * sizeof(fp8)));

	    prev_m = m;
	    prev_n = n;
	    prev_k = k;
    }

    if (!streamk_tmp) {
        HIP_CHECK(hipMalloc(&streamk_tmp, 7186 * 7186 * sizeof(fp32)));
        HIP_CHECK(hipMalloc(&streamk_counter, 7186 * 7186 * sizeof(fp32))); // whatever, its large enough
    }

    hipEvent_t start, tab, stop;
    if (measure) {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&tab));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start, 0));
    }

	if (!fused_transpose) {
	    transpose_faster_fused<<<
	        dim3(ceil_div(m, 64) + ceil_div(n, 64), ceil_div(k, 64)),
	        dim3(256)
	    >>>(tmp_aT, tmp_bT, a, b, m, n, k, ceil_div(m, 64));
	    HIP_CHECK(hipGetLastError());
	    if (measure) {
	        HIP_CHECK(hipEventRecord(tab, 0));
	    }
    }

	int streamk_splits = streamk ? 2 : 1;
	int streamk_stride = ceil_div(ceil_div(k, streamk_splits), group_size) * group_size;

	if (streamk) {
		HIP_CHECK(hipMemsetAsync(streamk_counter, 0, grid_m * grid_n * sizeof(int)));
	}

    kernel<
        block_size,
        block_tile,
        warp_tile,
        load_type,
        pipeline_depth,
        fused_transpose,
        streamk
    ><<<
        dim3(streamk_splits, grid_m, grid_n),
        dim3(block_size),
        0
    >>>(
        fused_transpose ? a : tmp_aT,
        fused_transpose ? b : tmp_bT,
        as,
        bs,
        c,
        m,
        n,
        k,
        streamk_stride,
        streamk_tmp,
	    streamk_counter
    );

    HIP_CHECK(hipGetLastError());

    if (measure) {
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));

        if (!fused_transpose) {
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, start, tab));
            const double time = ms / 1'000.0;
            const double memory = (m + n) * k * 2 * sizeof(fp8);
            const double memory_tput = memory / time / (1000.0 * 1000 * 1000);

            std::cerr
                << std::fixed
                << std::setprecision(2)
                << "transpose a, b: "
                << ms << " ms "
                << (time * 1'000'000) << " us "
                << memory_tput << " GB/s "
                << std::endl;
        }

        {
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, fused_transpose ? start : tab, stop));
            const double time = ms / 1'000.0;

            const double memory = m * k * sizeof(fp8) + n * k * sizeof(fp8) + m * n * sizeof(bf16);
            const double ops =
                double(m) * n * k * 2 // Multiply + add for each pair
                + (double(m) * n * 2) * ceil_div(k, warp_tile.k); // scaling

            const double memory_tput = memory / time / (1000.0 * 1000 * 1000);
            const double compute_tput = ops / time / (1000.0 * 1000 * 1000 * 1000);

            std::cerr
                << std::fixed
                << std::setprecision(2)
                << "mma: "
                << "(" << m << " " << n << " " << k << ") "
                << "(" << block_tile.m << " " << block_tile.n << " " << block_tile.k << ") "
                << "(" << warp_tile.m << " " << warp_tile.n << " " << warp_tile.k << ") "
                << pipeline_depth << " "
                << fused_transpose << " "
                << streamk << " "
                << block_size << ": "
                << ms << " ms "
                << (time * 1'000'000) << " us "
                << memory_tput << " GB/s "
                << compute_tput << " TFLOPS"
                << std::endl;
        }

        {
            float ms = 0;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            const double time = ms / 1'000.0;
            std::cerr
                << std::fixed
                << std::setprecision(2)
                << "total: "
                << ms << " ms "
                << (time * 1'000'000) << " us "
                << std::endl;
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(tab));
        HIP_CHECK(hipEventDestroy(stop));
    }
}

void run(
    uintptr_t a_ptr,
    uintptr_t b_ptr,
    uintptr_t as_ptr,
    uintptr_t bs_ptr,
    uintptr_t c_ptr,
    int m,
    int n,
    int k,
    bool measure
) {
    // In the kernels we assume 128-bit alignment.
    assert(a_ptr % 16 == 0);
    assert(b_ptr % 16 == 0);
    assert(as_ptr % 16 == 0);
    assert(bs_ptr % 16 == 0);
    assert(c_ptr % 16 == 0);

    const auto* a = reinterpret_cast<const fp8*>(a_ptr);
    const auto* b = reinterpret_cast<const fp8*>(b_ptr);
    const auto* as = reinterpret_cast<const fp32*>(as_ptr);
    const auto* bs = reinterpret_cast<const fp32*>(bs_ptr);
    auto* c = reinterpret_cast<bf16*>(c_ptr);

    /*
    //unroll<0, 2>([&]<int streamk> {
        unroll<0, 3>([&]<int p> {
            // unroll<0, 2>([&]<int fused_transpose> {
                unroll<0, 2>([&]<int tile_type> {
                    unroll<6, 10>([&]<int shift> {
                        constexpr bool sk = false;
                        constexpr bool t = true;
                        constexpr int z = 1 << shift;
                        constexpr cube tile = tile_type == 0 ? cube{32, 32, 16} : cube{16, 16, 32};

                        if (k < 256 && sk) {
                            return;
                        }

                        if constexpr (z <= 256) {
                            run_kernel<z, cube{64, 64, 64}, tile, p, t, sk>(a, b, as, bs, c, m, n, k, measure);

                            run_kernel<z, cube{64, 64, 128}, tile, p, t, sk>(a, b, as, bs, c, m, n, k, measure);
                            run_kernel<z, cube{64, 128, 64}, tile, p, t, sk>(a, b, as, bs, c, m, n, k, measure);
                            run_kernel<z, cube{128, 64, 64}, tile, p, t, sk>(a, b, as, bs, c, m, n, k, measure);

                            run_kernel<z, cube{128, 128, 64}, tile, p, t, sk>(a, b, as, bs, c, m, n, k, measure);
                            run_kernel<z, cube{64, 128, 128}, tile, p, t, sk>(a, b, as, bs, c, m, n, k, measure);
                            run_kernel<z, cube{128, 64, 128}, tile, p, t, sk>(a, b, as, bs, c, m, n, k, measure);
                        }

                        run_kernel<z, cube{128, 128, 128}, tile, p, t, sk>(a, b, as, bs, c, m, n, k, measure);

                        run_kernel<z, cube{256, 128, 128}, tile, p, t, sk>(a, b, as, bs, c, m, n, k, measure);
                        run_kernel<z, cube{128, 256, 128}, tile, p, t, sk>(a, b, as, bs, c, m, n, k, measure);
                    });
                });
            // });
        });
    // });
    */

    if (m == 1024 && n == 1536 && k == 7168) {
        run_kernel<256, cube{64, 128, 64}, cube{32, 32, 16}, 2, true, true>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 1024 && n == 3072 && k == 1536) {
        run_kernel<256, cube{128, 128, 64}, cube{16, 16, 32}, 2, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 1024 && n == 576 && k == 7168) {
        run_kernel<256, cube{64, 64, 128}, cube{32, 32, 16}, 2, true, true>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 1024 && n == 7168 && k == 256) {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 1, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 1024 && n == 7168 && k == 2048) {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 1, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 1024 && n == 4608 && k == 7168) {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 2, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 1024 && n == 7168 && k == 2304) {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 1, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 1024 && n == 512 && k == 7168) {
        run_kernel<256, cube{64, 64, 128}, cube{32, 32, 16}, 2, true, true>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 1024 && n == 4096 && k == 512) {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 2, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 6144 && n == 1536 && k == 7168) {
        run_kernel<256, cube{128, 128, 64}, cube{16, 16, 32}, 0, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 6144 && n == 3072 && k == 1536) {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 1, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 6144 && n == 576 && k == 7168) {
        run_kernel<256, cube{128, 128, 64}, cube{16, 16, 32}, 2, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 6144 && n == 7168 && k == 256) {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 1, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 6144 && n == 7168 && k == 2048) {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 1, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 6144 && n == 4608 && k == 7168) {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 1, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 6144 && n == 7168 && k == 2304) {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 1, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 6144 && n == 512 && k == 7168) {
        run_kernel<256, cube{128, 128, 64}, cube{16, 16, 32}, 2, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else if (m == 6144 && n == 4096 && k == 512) {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 1, true, false>(a, b, as, bs, c, m, n, k, measure);
    } else {
        run_kernel<256, cube{128, 128, 64}, cube{32, 32, 16}, 0, true, false>(a, b, as, bs, c, m, n, k, measure);
    }

    // run_kernel<64, cube{64, 64, 128}, cube{32, 32, 16}, true>(a, b, as, bs, c, m, n, k, measure);
}

PYBIND11_MODULE(fp8, m) {
  m.def("fp8", &run, "HIP kernel");
}
