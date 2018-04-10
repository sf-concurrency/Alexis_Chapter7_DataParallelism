//
//  main.swift
//  MultiplyArrays
//
//  Created by Alexis Gallagher on 4/10/18.
//  Copyright © 2018 Topology Eyewear. All rights reserved.
//

import Foundation
import OpenCL

print("Hello, World!")

let NUM_ELEMENTS:Int = 1024

let openclProgramSource:String = """
__kernel void multiply_arrays(__global const float* inputA,
__global const float* inputB,
__global float* output) {

int i = get_global_id(0);
output[i] = inputA[i] * inputB[i];
}
"""

func random_fill( a:inout Array<cl_float>) -> Void {
  for i in 0..<a.count {
    a[i] = Float(arc4random()) / Float(UINT32_MAX)
  }
}

@discardableResult
func main() -> UInt8
{
  // get a handle to the platform
  var platform:cl_platform_id? = nil
  clGetPlatformIDs(1, &platform, nil)

  // specify the CPU device on that platform
  var device:cl_device_id? = nil
  clGetDeviceIDs(platform, UInt64(CL_DEVICE_TYPE_CPU), 1, &device, nil)

  // create ... a context, a queue, a program
  let context = clCreateContext(nil, 1, &device, nil, nil, nil)

  let queue = clCreateCommandQueue(context, device, 0, nil)

  let program:cl_program? = openclProgramSource.withCString { (ptr:UnsafePointer<Int8>) -> cl_program? in
    var wrappedPtr:UnsafePointer<Int8>? = ptr
    return withUnsafeMutablePointer(to: &wrappedPtr, { (ptrptr:UnsafeMutablePointer<UnsafePointer<Int8>?>) -> cl_program? in

      // we create a program object using source code and a context
      // all this ceremony is to interop with pointery C types for the OpenCL API
      let result = clCreateProgramWithSource(context, 1, ptrptr, nil, nil)
      return result
    })
  }

  // build (compile?) the program
  clBuildProgram(program, 0, nil, nil, nil, nil)
  let kernel:cl_kernel = clCreateKernel(program, "multiply_arrays", nil)

  // define our input arrays in Swift and fill them with random floats
  var a = Array<cl_float>(repeating: 0, count: NUM_ELEMENTS)
  var b = Array<cl_float>(repeating: 0, count: NUM_ELEMENTS)

  random_fill(a: &a)
  random_fill(a: &b)

  // create an OpenCL buffer for input array A
  var input_a:cl_mem = a.withUnsafeMutableBytes { (p:UnsafeMutableRawBufferPointer) -> cl_mem in
    return clCreateBuffer(context,
                   UInt64(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR),
                   MemoryLayout<cl_float>.stride * NUM_ELEMENTS,
                   p.baseAddress, nil)
  }

  // create an OpenCL buffer for input array A
  var input_b:cl_mem = b.withUnsafeMutableBytes { (p:UnsafeMutableRawBufferPointer) -> cl_mem in
    return clCreateBuffer(context,
                          UInt64(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR),
                          MemoryLayout<cl_float>.stride * NUM_ELEMENTS,
                          p.baseAddress, nil)
  }

  // create an OpenCL buffer for our output array
  var output:cl_mem = clCreateBuffer(context,
                                     UInt64(CL_MEM_WRITE_ONLY),
                                     MemoryLayout<cl_float>.stride * NUM_ELEMENTS,
                                     nil, nil)

  // specify that these buffers are the "arguments" to the OpenCL "kernel" from our OpenCL "program"
  clSetKernelArg(kernel, 0, MemoryLayout<cl_mem>.size, &input_a)
  clSetKernelArg(kernel, 1, MemoryLayout<cl_mem>.size, &input_b)
  clSetKernelArg(kernel, 2, MemoryLayout<cl_mem>.size, &output)

  // COMPUTE: this tells openCL to put the "kernel" (the relevant bit of our OpenCL "program")
  // onto the OpenCL queue.
  //
  // work_units is an instruction about how much to parallelize...
  var work_units = NUM_ELEMENTS
  clEnqueueNDRangeKernel(queue, kernel, cl_uint(CL_TRUE), nil, &work_units, nil, 0, nil, nil)

  // define an output array in Swift
  var results = Array<cl_float>(repeating: 1, count: NUM_ELEMENTS)
  // fill our Swift output array with values from our OpenCL buffer
  results.withUnsafeMutableBytes {
    (p:UnsafeMutableRawBufferPointer) -> Void in
    clEnqueueReadBuffer(queue, output, cl_uint(CL_TRUE), 0,
                        MemoryLayout<cl_float>.stride * NUM_ELEMENTS,
                        p.baseAddress, 0, nil, nil)
  }

  // release all the OpenCL structures: inputs, outputs, the kernel, the program, the queue, the context
  clReleaseMemObject(input_a)
  clReleaseMemObject(input_b)
  clReleaseMemObject(output)
  clReleaseKernel(kernel)
  clReleaseProgram(program)
  clReleaseCommandQueue(queue)
  clReleaseContext(context)

  // print our Swift input arrays and output array
  for i in 0..<NUM_ELEMENTS {
    print("\(a[i]) * \(b[i]) = \(results[i])")
  }

  return 0
}

main()

