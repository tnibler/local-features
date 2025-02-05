#ifndef _PIXEL_TYPE_GLSL
#define _PIXEL_TYPE_GLSL

#if defined(PRECISION_FLOAT32) && !defined(PRECISION_FLOAT16)
#elif !defined(PRECISION_FLOAT32) && defined(PRECISION_FLOAT16)
#error FIXME: float16 disabled
#else
#error Must define exactly one of PRECISION_FLOAT32 or PRECISION_FLOAT16
#endif


#ifdef PRECISION_FLOAT32
    #define IMG_FORMAT r32f
    #define PX_TYPE float
    #define vko_image2DArray_FORMAT vko_image2DArray_r32f
    #define vec4_FORMAT vec4
#endif
#ifdef PRECISION_FLOAT16
    #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
    #extension GL_EXT_shader_16bit_storage : require

    #define IMG_FORMAT r16f
    #define PX_TYPE float16_t
    #define vko_image2DArray_FORMAT vko_image2DArray_r16f
    #define vec4_FORMAT f16vec4

    precision mediump float;
#endif 

layout(buffer_reference, std430, buffer_reference_align=8) buffer PixelBuffer {
    PX_TYPE[] data;
};

#endif // _PIXEL_TYPE_GLSL
