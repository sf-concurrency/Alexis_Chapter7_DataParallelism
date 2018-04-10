

# Chapter 7, Data Parallelism, Notes

## Reading notes

- Is OpenCL still a thing?
- Is OpenCL accessible via Swift on iOS?

Seems to be available on macOS: https://developer.apple.com/opencl/

Seems NOT to be available on iOS: https://stackoverflow.com/a/28272333/577888

2-year old codebases on Swift/OpenCL:
- https://github.com/damienpontifex/SwiftOpenCL
- https://github.com/damienpontifex/opencl-in-action-swift

Apple Sample code is all C of course:
- https://developer.apple.com/library/content/samplecode/OpenCL_Parallel_Prefix_Sum_Example/Introduction/Intro.html#//apple_ref/doc/uid/DTS40008183
- https://developer.apple.com/library/content/documentation/Performance/Conceptual/OpenCL_MacProgGuide/Introduction/Introduction.html#//apple_ref/doc/uid/TP40008312

## Day 1: GPGPU programming

How much of this has been superceded by the use of libraries like
numpy?

Review article on numpy and GPU:
http://stsievert.com/blog/2016/07/01/numpy-gpu/

Seems like there are variants that give you GPU but by default you
just get vectorized operations on the CPU.

Key graph:

> An OpenCL context represents an environment within which OpenCL
> kernels can execute. To create a context, we first need to identify
> the platform that we want to use and which devices within that
> platform we want to execute our kernel

So:
- a "kernel" is an OpenCL program
- you create a "context" defines where you execute
- with the context, you create a "queue" where you can send kernel programs
- you need to compile the kernel program first, by actually delivering a pointer to the kernel as data.
- use special functions to move data into buffers a compiled kernel can access
- 

Reminiscent of TensorFlow

### Day 1 self-study

Day 1 Self-Study
Find
• The OpenCL specification

https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf

• The OpenCL API reference card

https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf


• The language used to define an OpenCL kernel is C-like. How does it differ from C?

Says the spec:

> This section describes the OpenCL C programming language used to
> create kernels that are executed on OpenCL device(s). The OpenCL C
> programming language (also referred to as OpenCL C) is based on the
> ISO/IEC 9899:1999 C language specification (a.k.a. C99
> specification) with specific extensions and restrictions. Please
> refer to the ISO/IEC 9899:1999 specification for a detailed
> description of the language grammar. This section describes
> modifications and restrictions to ISO/IEC 9899:1999 supported in
> OpenCL C.

So: based on C99, with "extension and restrictions".


[Says wikipedia](https://en.wikipedia.org/wiki/OpenCL#OpenCL_C_language):

> The programming language that is used to write compute kernels is
> called OpenCL C and is based on C99,[16] but adapted to fit the
> device model in OpenCL. Memory buffers reside in specific levels of
> the memory hierarchy, and pointers are annotated with the region
> qualifiers __global, __local, __constant, and __private, reflecting
> this. Instead of a device program having a main function, OpenCL C
> functions are marked __kernel to signal that they are entry points
> into the program to be called from the host program. Function
> pointers, bit fields and variable-length arrays are omitted,
> recursion is forbidden.[17] The C standard library is replaced by a
> custom set of standard functions, geared toward math programming.
> 
> OpenCL C is extended to facilitate use of parallelism with vector
> types and operations, synchronization, and functions to work with
> work-items and work-groups.[17] In particular, besides scalar types
> such as float and double, which behave similarly to the
> corresponding types in C, OpenCL provides fixed-length vector types
> such as float4 (4-vector of single-precision floats); such vector
> types are available in lengths two, three, four, eight and sixteen
> for various base types.[16]:§ 6.1.2 Vectorized operations on these
> types are intended to map onto SIMD instructions sets, e.g., SSE or
> VMX, when running OpenCL programs on CPUs.[12] Other specialized
> types include 2-d and 3-d image types.


So to summarize, the changes to base C are:
- attributes to specify location of storage
- no function pointers, bit fields, and variable length arrays
- vector data types
- image buffer data types
- no stdlib, instead more mathy stdlib
- more synchronization primitives

Do:

- Modify our array multiplication kernel to deal with arrays of
  different types, and profile the resulting performance. How does it
  vary with data type? Does the size (in bytes) of the data type have
  any bearing on performance, both in absolute terms and in comparison
  to CPU performance?

- We created and initialized our buffers by passing
  `CL_MEM_COPY_HOST_PTR` to `clCreateBuffer()`. Rewrite the host to use
  `CL_MEM_USE_HOST_PTR` or `CL_MEM_ALLOC_HOST_PTR` (you will need to do
  more than just change the flag for the code to remain functional),
  and benchmark the resulting performance. What are the trade-offs
  between different buffer-allocation strategies?

 
- Rewrite the host to use `clEnqueueMapBuffer()` instead of
  `clCreateBuffer()` and profile the result. When might this be an
  appropriate choice? When might it not?

- The OpenCL language provides a number of data types over and above
  those provided by standard C—in particular, it includes vector types
  such as float4 or ulong3. Rewrite our kernel to multiply two buffers
  of vectors. How are these vector types represented on the host?


