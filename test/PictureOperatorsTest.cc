/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "gtest/gtest.h"
#include "EbDefinitions.h"
#include "EbPictureOperators.h"
#include "EbUnitTestUtility.h"
#include "EbUnitTest.h"

#ifndef NON_AVX512_SUPPORT
static const int num_test = 27;

EbFullDistortionKernelCbfZero32Bits full_distortion_kernel_cbf_zero32_bits_funcptr_array[ASM_TYPE_TOTAL] = {
    full_distortion_kernel_cbf_zero32_bits_avx2,
    full_distortion_kernel_cbf_zero32_bits_avx512,
};

EbFullDistortionKernel32Bits full_distortion_kernel32_bits_funcptr_array[ASM_TYPE_TOTAL] = {
    full_distortion_kernel32_bits_avx2,
    full_distortion_kernel32_bits_avx512,
};

struct DistInfo {
    uint32_t width;
    uint32_t height;
};

static const struct DistInfo size_info[num_test] = {
    { 4,  4},
    { 4,  8},
    { 4, 16},
    { 8,  4},
    { 8,  8},
    { 8, 16},
    { 8, 32},
    {16,  4},
    {16,  8},
    {16, 12},
    {16, 16},
    {16, 32},
    {16, 64},
    {32,  8},
    {32, 16},
    {32, 24},
    {32, 32},
    {32, 64},
    {32, 128},
    {64, 16},
    {64, 32},
    {64, 48},
    {64, 64},
    {64, 128},
    {128, 32},
    {128, 64},
    {128, 128}
};

static void init_data(uint8_t **input, uint32_t *input_stride, uint8_t **recon,
    uint32_t *recon_stride)
{
    *input_stride = eb_create_random_aligned_stride(MAX_SB_SIZE, 64);
    *recon_stride = eb_create_random_aligned_stride(MAX_SB_SIZE, 64);
    *input = (uint8_t*)malloc(sizeof(**input) * MAX_SB_SIZE * *input_stride);
    *recon = (uint8_t*)malloc(sizeof(**recon) * MAX_SB_SIZE * *recon_stride);
    eb_buf_random_u8(*input, MAX_SB_SIZE * *input_stride);
    eb_buf_random_u8(*recon, MAX_SB_SIZE * *recon_stride);
}

static void uninit_data(uint8_t *input, uint8_t *recon)
{
    free(input);
    free(recon);
}

void init_data_for_distortion_test(int32_t **coeff, uint32_t *coeff_stride, int32_t ** recon_coeff, uint32_t *recon_coeff_stride) {
    *coeff_stride = eb_create_random_aligned_stride(MAX_SB_SIZE, 64);
    *recon_coeff_stride = eb_create_random_aligned_stride(MAX_SB_SIZE, 64);
    TEST_ALLIGN_MALLOC(int32_t*, *coeff, sizeof(int32_t) * MAX_SB_SIZE * *coeff_stride);
    TEST_ALLIGN_MALLOC(int32_t*, *recon_coeff, sizeof(int32_t) * MAX_SB_SIZE * *recon_coeff_stride);
    eb_buf_random_s32(*coeff, MAX_SB_SIZE * *coeff_stride);
    eb_buf_random_s32(*recon_coeff, MAX_SB_SIZE * *recon_coeff_stride);
}

static void uninit_data2(int32_t *coeff, int32_t *recon_coeff) {
    TEST_ALLIGN_FREE(coeff);
    TEST_ALLIGN_FREE(recon_coeff);
}

TEST(PictureOperators, full_distortion_cbf_zero32bits)
{
    int32_t  *coeff, *recon_coeff;
    uint32_t   coeff_stride, recon_coeff_stride;

    for (int i = 0; i < 10; i++) {
        init_data_for_distortion_test(&coeff, &coeff_stride, &recon_coeff, &recon_coeff_stride);

        for (int j = 0; j < num_test; j++) {
            const uint32_t area_width = size_info[j].width;
            const uint32_t area_height = size_info[j].height;
            uint64_t   distortion_result_base[DIST_CALC_TOTAL] = { 0, 0 };
            uint64_t   distortion_result_opt[DIST_CALC_TOTAL] = { 0, 0 };
            full_distortion_kernel_cbf_zero32_bits_funcptr_array[0](coeff, coeff_stride, recon_coeff, recon_coeff_stride, distortion_result_base, area_width, area_height);
            full_distortion_kernel_cbf_zero32_bits_funcptr_array[1](coeff, coeff_stride, recon_coeff, recon_coeff_stride, distortion_result_opt, area_width, area_height);
            EXPECT_EQ(distortion_result_base[0], distortion_result_opt[0]);
            EXPECT_EQ(distortion_result_base[1], distortion_result_opt[1]);
        }

        uninit_data2(coeff, recon_coeff);
    }
}

TEST(PictureOperators, full_distortion_32bits)
{
    int32_t  *coeff, *recon_coeff;
    uint32_t   coeff_stride, recon_coeff_stride;

    for (int i = 0; i < 10; i++) {
        init_data_for_distortion_test(&coeff, &coeff_stride, &recon_coeff, &recon_coeff_stride);

        for (int j = 0; j < num_test; j++) {
            const uint32_t area_width = size_info[j].width;
            const uint32_t area_height = size_info[j].height;
            uint64_t   distortion_result_base[DIST_CALC_TOTAL] = { 0, 0 };
            uint64_t   distortion_result_opt[DIST_CALC_TOTAL] = { 0, 0 };
            full_distortion_kernel32_bits_funcptr_array[0](coeff, coeff_stride, recon_coeff, recon_coeff_stride, distortion_result_base, area_width, area_height);
            full_distortion_kernel32_bits_funcptr_array[1](coeff, coeff_stride, recon_coeff, recon_coeff_stride, distortion_result_opt, area_width, area_height);
            EXPECT_EQ(distortion_result_base[0], distortion_result_opt[0]);
            EXPECT_EQ(distortion_result_base[1], distortion_result_opt[1]);
        }

        uninit_data2(coeff, recon_coeff);
    }
}

#endif
