/* vec.c
 *
 * Author:
 * Date  :
 *
 *  Description
 */

/* Standard C includes */
#include <stdlib.h>
#include <printf.h>
#include "arm_neon.h"


/* Include common headers */
#include "common/macros.h"
#include "common/types.h"
#include "common/vmath.h"

/* Include application-specific headers */
#include "include/types.h"

#define inv_sqrt_2xPI 0.39894228040143270286

float32x4_t CNDF(float32x4_t InputX) {
    const float32x4_t vInvSqrt2xPI = vdupq_n_f32(inv_sqrt_2xPI);

    float32x4_t vSign = vcltq_f32(InputX, vdupq_n_f32(0.0));
    float32x4_t vInputX = vabsq_f32(InputX);

    float32x4_t xK2, xK2_2, xK2_3, xK2_4, xK2_5;
    float32x4_t xLocal, xLocal_1, xLocal_2, xLocal_3;

    float32x4_t vNPrimeofX = vexp_f32(
            vmulq_f32(
                    vdupq_n_f32(-0.5),
                    vmulq_f32(vInputX, vInputX)
            )
    );
    vNPrimeofX = vmulq_f32(vNPrimeofX, vInvSqrt2xPI);

    xK2 = vmulq_f32(vdupq_n_f32(0.2316419), vInputX);
    xK2 = vaddq_f32(vdupq_n_f32(1.0), xK2);
    xK2 = vdivq_f32(vdupq_n_f32(1.0), xK2);
    xK2_2 = vmulq_f32(xK2, xK2);
    xK2_3 = vmulq_f32(xK2_2, xK2);
    xK2_4 = vmulq_f32(xK2_3, xK2);
    xK2_5 = vmulq_f32(xK2_4, xK2);

    xLocal_1 = vmulq_f32(xK2, vdupq_n_f32(0.319381530));
    xLocal_2 = vmulq_f32(xK2_2, vdupq_n_f32(-0.356563782));
    xLocal_3 = vmulq_f32(xK2_3, vdupq_n_f32(1.781477937));
    xLocal_2 = vaddq_f32(xLocal_2, xLocal_3);
    xLocal_3 = vmulq_f32(xK2_4, vdupq_n_f32(-1.821255978));
    xLocal_2 = vaddq_f32(xLocal_2, xLocal_3);
    xLocal_3 = vmulq_f32(xK2_5, vdupq_n_f32(1.330274429));
    xLocal_2 = vaddq_f32(xLocal_2, xLocal_3);

    xLocal_1 = vaddq_f32(xLocal_2, xLocal_1);
    xLocal = vmulq_f32(xLocal_1, vNPrimeofX);
    xLocal = vsubq_f32(vdupq_n_f32(1.0), xLocal);

    float32x4_t signedOutput = vsubq_f32(vdupq_n_f32(1.0), xLocal);

    return vbslq_f32(vSign, signedOutput, xLocal);;
}


float32x4_t
blackScholes(float32x4_t vSptPrice, float32x4_t vStrike,
             float32x4_t vRate, float32x4_t vVolatility,
             float32x4_t vOtime, uint32x4_t vOType) {

    float32x4_t vSqrtTime = vsqrtq_f32(vOtime);
    float32x4_t vLogTerm = vlog_f32(
            vdivq_f32(vSptPrice, vStrike)
    );

    float32x4_t vPowerTerm = vmulq_f32(vVolatility, vVolatility);
    vPowerTerm = vmulq_f32(vPowerTerm, vdupq_n_f32(0.5));

    float32x4_t xD1 = vaddq_f32(vRate, vPowerTerm);
    xD1 = vmulq_f32(xD1, vOtime);
    xD1 = vaddq_f32(xD1, vLogTerm);

    float32x4_t vDen = vmulq_f32(vVolatility, vSqrtTime);
    xD1 = vdivq_f32(xD1, vDen);
    float32x4_t xD2 = vsubq_f32(xD1, vDen);

    float32x4_t vCDFd1 = CNDF(xD1);
    float32x4_t vCDFd2 = CNDF(xD2);

    float32x4_t vFutureValueX = vmulq_f32(
            vStrike,
            vexp_f32(
                    vmulq_f32(
                            vmulq_f32(vdupq_n_f32(-1), vRate),
                            vOtime
                    )
            )
    );

    float32x4_t vOptionPriceOTypeC = vsubq_f32(
            vmulq_f32(vSptPrice, vCDFd1),
            vmulq_f32(vFutureValueX, vCDFd2)
    );

    float32x4_t vOptionPriceOTypeP = vsubq_f32(
            vmulq_f32(vFutureValueX, vsubq_f32(vdupq_n_f32(1), vCDFd2)),
            vmulq_f32(vSptPrice, vsubq_f32(vdupq_n_f32(1), vCDFd1))
    );

    int32x4_t v67 = vdupq_n_s32(67);

    float32x4_t vOptionPrice = vbslq_f32(
            vceqq_s32(vOType, v67),
            vOptionPriceOTypeC,
            vOptionPriceOTypeP
    );

    return vOptionPrice;
}

void blackScholesHelper(float *sptprice, float *strike, float *rate,
                        float *volatility, float *otime, char *otype,
                        float *output, size_t numOptions) {

    int i;
    float32x4_t vSptPrice, vStrike, vRate, vVolatility, vOtime;
    int32x4_t vOType;

    for (i = 0; i < numOptions; i += 4) {
        int remainingOptions = numOptions - i;
        if (remainingOptions > 4) remainingOptions = 4;

        // Load data
        vSptPrice = vld1q_f32(sptprice + i);
        vStrike = vld1q_f32(strike + i);
        vRate = vld1q_f32(rate + i);
        vVolatility = vld1q_f32(volatility + i);
        vOtime = vld1q_f32(otime + i);

        uint8x16_t vOTypeChar = vld1q_u8(otype + i);
        uint16x8_t vOTypeHalf = vmovl_u8(vget_low_u8(vOTypeChar));
        vOType = vmovl_u16(vget_low_u16(vOTypeHalf));

        // Compute option prices
        float32x4_t vOptionPrices = blackScholes(vSptPrice, vStrike, vRate, vVolatility, vOtime, vOType);

        // Temporarily store the results in an array
        float temp[4];
        vst1q_f32(temp, vOptionPrices);

        // Copy the needed results to the output
        for (int j = 0; j < remainingOptions; j++) {
            output[i + j] = temp[j];
        }


    }

}

/* Alternative Implementation */
void *
impl_vector(void *args) {
    /* Get the argument struct */
    args_t *parsed_args = (args_t *) args;

    blackScholesHelper(
            parsed_args->sptPrice,
            parsed_args->strike,
            parsed_args->rate,
            parsed_args->volatility,
            parsed_args->otime,
            parsed_args->otype,
            parsed_args->output,
            parsed_args->num_stocks
    );
}


