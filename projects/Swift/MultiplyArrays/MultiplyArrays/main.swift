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
func main() -> UInt8 {
  var platform:cl_platform_id? = nil
  clGetPlatformIDs(1, &platform, nil)

  var device:cl_device_id? = nil
  clGetDeviceIDs(platform, UInt64(CL_DEVICE_TYPE_CPU), 1, &device, nil)

  let context = clCreateContext(nil, 1, &device, nil, nil, nil)

  let queue = clCreateCommandQueue(context, device, 0, nil)

  let program:cl_program? = openclProgramSource.withCString { (ptr:UnsafePointer<Int8>) -> cl_program? in
    var wrappedPtr:UnsafePointer<Int8>? = ptr
    return withUnsafeMutablePointer(to: &wrappedPtr, { (ptrptr:UnsafeMutablePointer<UnsafePointer<Int8>?>) -> cl_program? in

      let result = clCreateProgramWithSource(context, 1, ptrptr, nil, nil)
      return result
    })
  }

  clBuildProgram(program, 0, nil, nil, nil, nil)

  let kernel:cl_kernel = clCreateKernel(program, "multiply_arrays", nil)

  var a = Array<cl_float>(repeating: 0, count: NUM_ELEMENTS)
  var b = Array<cl_float>(repeating: 0, count: NUM_ELEMENTS)

  random_fill(a: &a)
  random_fill(a: &b)

  // todo: ranndomize

  var input_a:cl_mem = a.withUnsafeMutableBytes { (p:UnsafeMutableRawBufferPointer) -> cl_mem in
    return clCreateBuffer(context,
                   UInt64(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR),
                   MemoryLayout<cl_float>.stride * NUM_ELEMENTS,
                   p.baseAddress, nil)
  }
  var input_b:cl_mem = b.withUnsafeMutableBytes { (p:UnsafeMutableRawBufferPointer) -> cl_mem in
    return clCreateBuffer(context,
                          UInt64(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR),
                          MemoryLayout<cl_float>.stride * NUM_ELEMENTS,
                          p.baseAddress, nil)
  }


  var output:cl_mem = clCreateBuffer(context,
                                     UInt64(CL_MEM_WRITE_ONLY),
                                     MemoryLayout<cl_float>.stride * NUM_ELEMENTS,
                                     nil, nil)

  clSetKernelArg(kernel, 0, MemoryLayout<cl_mem>.size, &input_a)
  clSetKernelArg(kernel, 1, MemoryLayout<cl_mem>.size, &input_b)
  clSetKernelArg(kernel, 2, MemoryLayout<cl_mem>.size, &output)

  var work_units = NUM_ELEMENTS
  clEnqueueNDRangeKernel(queue, kernel, cl_uint(CL_TRUE), nil, &work_units, nil, 0, nil, nil)

  var results = Array<cl_float>(repeating: 1, count: NUM_ELEMENTS)

  results.withUnsafeMutableBytes {
    (p:UnsafeMutableRawBufferPointer) -> Void in
    clEnqueueReadBuffer(queue, output, cl_uint(CL_TRUE), 0,
                        MemoryLayout<cl_float>.stride * NUM_ELEMENTS,
                        p.baseAddress, 0, nil, nil)
  }

  clReleaseMemObject(input_a)
  clReleaseMemObject(input_b)
  clReleaseMemObject(output)
  clReleaseKernel(kernel)
  clReleaseProgram(program)
  clReleaseCommandQueue(queue)
  clReleaseContext(context)

  for i in 0..<NUM_ELEMENTS {
    print("\(a[i]) * \(b[i]) = \(results[i])")
  }

  return 0
}

main()

